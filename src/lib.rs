use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyBool, PySet, PyString},
};
use rayon::prelude::*;
use std::collections::HashSet;
use std::sync::{Arc, RwLock};

#[pyclass(name = "AhoMatcher")]
struct AhoMatcher {
    ac_impl: Option<Arc<AhoCorasick>>,
    // 存储模式字符串的向量
    patterns: Arc<Vec<String>>,
    // 存储模式组件的向量，分为包含组和排除组
    pattern_components: Arc<Vec<(Vec<String>, Vec<Vec<String>>)>>,
    // 是否使用逻辑匹配
    use_logic: bool,
}

#[pymethods]
impl AhoMatcher {
    #[new]
    #[pyo3(signature = (use_logic=None))]
    fn new(use_logic: Option<&Bound<'_, PyBool>>) -> PyResult<Self> {
        // 如果提供了 use_logic 参数，则使用该参数的值，否则默认为 true
        let use_logic_value = match use_logic {
            Some(value) => value.is_true(),
            None => true,
        };

        Ok(Self {
            ac_impl: None,
            patterns: Arc::new(Vec::new()),
            pattern_components: Arc::new(Vec::new()),
            use_logic: use_logic_value,
        })
    }

    #[pyo3(text_signature = "(patterns: set)")]
    fn build(&mut self, py: Python<'_>, patterns: &Bound<'_, PySet>) -> PyResult<()> {
        // 获取模式的数量
        let pattern_count = patterns.len();
        // 初始化存储有效模式的向量，预估大小为模式数量的两倍
        let mut valid_patterns = Vec::with_capacity(pattern_count * 2);
        // 初始化存储原始模式的向量
        let mut original_patterns = Vec::with_capacity(pattern_count);
        // 初始化存储模式组件的向量
        let mut pattern_components = Vec::with_capacity(pattern_count);

        // 从 Python 中收集模式
        let pattern_vec: Vec<String> = patterns
            .iter()
            .map(|pat| pat.extract::<&str>().map(String::from))
            .collect::<PyResult<Vec<_>>>()?;

        // 并行处理模式，不持有GIL
        let processed = py.allow_threads(|| {
            pattern_vec
                .into_par_iter()
                .map(|pattern| {
                    if pattern.is_empty() {
                        return Err(PyValueError::new_err("Pattern cannot be empty"));
                    }

                    let orig_pattern = pattern.clone();
                    
                    if self.use_logic {
                        // 使用～分割包含和排除部分
                        let mut parts = pattern.splitn(2, '~');
                        let positive_part = parts.next().unwrap_or("");
                        
                        // 处理包含组
                        let positive_terms: Vec<String> = positive_part
                            .split('&')
                            .map(str::trim)
                            .filter(|s| !s.is_empty())
                            .map(String::from)
                            .collect();

                        if positive_terms.is_empty() {
                            return Err(PyValueError::new_err(
                                "Pattern must contain at least one positive term before '~'",
                            ));
                        }

                        // 如果存在，处理排除组
                        let negative_term_groups = if let Some(negative_part) = parts.next() {
                            negative_part
                                .split('~')
                                .map(|segment| {
                                    segment
                                        .split('&')
                                        .map(str::trim)
                                        .filter(|s| !s.is_empty())
                                        .map(String::from)
                                        .collect::<Vec<String>>()
                                })
                                .filter(|group: &Vec<String>| !group.is_empty())
                                .collect::<Vec<Vec<String>>>()
                        } else {
                            Vec::new()
                        };

                        // 收集所有有效模式
                        let mut valid = positive_terms.clone();
                        for group in &negative_term_groups {
                            valid.extend(group.iter().cloned());
                        }
                        
                        Ok((orig_pattern, valid, (positive_terms, negative_term_groups)))
                    } else {
                        // 不使用逻辑匹配
                        Ok((
                            orig_pattern.clone(),
                            vec![orig_pattern.clone()],
                            (vec![orig_pattern], Vec::new()),
                        ))
                    }
                })
                .collect::<PyResult<Vec<_>>>()
        })?;

        // 收集结果
        for (orig, valid, comp) in processed {
            original_patterns.push(orig);
            valid_patterns.extend(valid);
            pattern_components.push(comp);
        }

        // 在GIL之外构建AhoCorasick
        let ac = py.allow_threads(|| {
            AhoCorasickBuilder::new()
                .match_kind(MatchKind::LeftmostLongest)
                .build(&valid_patterns)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;

        self.ac_impl = Some(Arc::new(ac));
        self.patterns = Arc::new(original_patterns);
        self.pattern_components = Arc::new(pattern_components);

        Ok(())
    }

    #[pyo3(text_signature = "(haystack: str)")]
    fn find<'py>(self_: PyRef<'_, Self>, haystack: &'py str) -> PyResult<Py<PySet>> {
        let py = self_.py();
        
        let ac_impl = match &self_.ac_impl {
            Some(ac) => Arc::clone(ac),
            None => return Err(PyValueError::new_err("AhoCorasick not built. Call build() first.")),
        };

        let patterns = Arc::clone(&self_.patterns);
        let components = Arc::clone(&self_.pattern_components);
        let use_logic = self_.use_logic;

        // 在GIL之外查找匹配项
        let matched_words = py.allow_threads(move || {
            let mut matches_set = HashSet::with_capacity(haystack.len() / 16);
            
            let matches = ac_impl
                .try_find_iter(haystack.as_bytes())
                .expect("Aho-Corasick matching failed");
                
            for m in matches {
                matches_set.insert(&haystack[m.start()..m.end()]);
            }
            
            matches_set
        });

        // 基于逻辑匹配处理结果
        let result_set = py.allow_threads(move || {
            if use_logic {
                let result = RwLock::new(HashSet::with_capacity(patterns.len()));

                components
                    .par_iter()
                    .enumerate()
                    .for_each(|(i, (pos_terms, neg_groups))| {
                        // 检查包含组
                        let all_positive = pos_terms
                            .iter()
                            .all(|term| matched_words.contains(term.as_str()));

                        // 检查包含组是否全部匹配
                        if !all_positive {
                            return;
                        }

                        // 检查排除组是否有完整匹配
                        let has_negative_match = neg_groups
                            .iter()
                            .any(|group| {
                                group
                                    .iter()
                                    .all(|term| matched_words.contains(term.as_str()))
                            });
                        
                        // 如果没有排除组匹配，则添加结果
                        if !has_negative_match {
                            let mut lock = result.write().unwrap();
                            lock.insert(patterns[i].clone());
                        }
                    });

                result.into_inner().unwrap()
            } else {
                // 不使用逻辑匹配
                patterns
                    .iter()
                    .filter(|pattern| matched_words.contains(pattern.as_str()))
                    .cloned()
                    .collect::<HashSet<_>>()
            }
        });

        // 将结果转换为pyset
        let py_set = PySet::new(py, result_set.iter().map(|s| PyString::new(py, s)))?;
        Ok(py_set.into())
    }
}

#[pymodule]
fn kwmatcher(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AhoMatcher>()?;
    Ok(())
}
