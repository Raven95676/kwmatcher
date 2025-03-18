use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyBool, PySet, PyString},
};
use rayon::prelude::*;
use std::collections::HashSet;
use std::sync::RwLock;

#[pyclass(name = "AhoMatcher")]
struct AhoMatcher {
    ac_impl: Option<AhoCorasick>,
    patterns: Vec<String>,
    pattern_components: Vec<(Vec<String>, Vec<Vec<String>>)>,
    use_logic: bool,
}

#[pymethods]
impl AhoMatcher {
    #[new]
    #[pyo3(signature = (use_logic=None))]
    fn new(use_logic: Option<&Bound<'_, PyBool>>) -> PyResult<Self> {
        let use_logic_value = match use_logic {
            Some(value) => value.is_true(),
            None => true,
        };

        Ok(Self {
            ac_impl: None,
            patterns: Vec::new(),
            pattern_components: Vec::new(),
            use_logic: use_logic_value,
        })
    }

    #[pyo3(text_signature = "(patterns: set)")]
    fn build(&mut self, py: Python<'_>, patterns: &Bound<'_, PySet>) -> PyResult<()> {
        let pattern_count = patterns.len();
        let mut valid_patterns = Vec::with_capacity(pattern_count);
        let mut original_patterns = Vec::with_capacity(pattern_count);
        let mut pattern_components = Vec::with_capacity(pattern_count);

        let pattern_vec: Vec<String> = patterns
            .iter()
            .map(|pat| pat.extract::<&str>().map(String::from))
            .collect::<PyResult<Vec<_>>>()?;

        let processed = py.allow_threads(|| {
            pattern_vec
                .into_par_iter()
                .map(|pattern| {
                    if pattern.is_empty() {
                        return Err(PyValueError::new_err("Pattern cannot be empty"));
                    }

                    let orig_pattern = pattern.clone();
                    let (valid_pats, components) = if self.use_logic {
                        let mut segments = pattern.split('~');
                        let positive_part = segments.next().unwrap_or("");

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

                        let negative_term_groups: Vec<Vec<String>> = segments
                            .map(|segment| {
                                segment
                                    .split('&')
                                    .map(str::trim)
                                    .filter(|s| !s.is_empty())
                                    .map(String::from)
                                    .collect()
                            })
                            .filter(|group: &Vec<String>| !group.is_empty())
                            .collect();

                        let mut valid = positive_terms.clone();
                        negative_term_groups
                            .iter()
                            .for_each(|group| valid.extend(group.iter().cloned()));
                        (valid, (positive_terms, negative_term_groups))
                    } else {
                        (vec![pattern.clone()], (vec![pattern], vec![]))
                    };

                    Ok((orig_pattern, valid_pats, components))
                })
                .collect::<PyResult<Vec<_>>>()
        })?;

        for (orig, valid, comp) in processed {
            original_patterns.push(orig);
            valid_patterns.extend(valid);
            pattern_components.push(comp);
        }

        self.ac_impl = Some(py.allow_threads(|| {
            AhoCorasickBuilder::new()
                .match_kind(MatchKind::LeftmostLongest)
                .build(&valid_patterns)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?);
        self.patterns = original_patterns;
        self.pattern_components = pattern_components;

        Ok(())
    }

    #[pyo3(text_signature = "(haystack: str)")]
    fn find(self_: PyRef<'_, Self>, haystack: &str) -> PyResult<Py<PySet>> {
        let ac_impl = self_
            .ac_impl
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("AhoCorasick not built. Call build() first."))?;

        let py = self_.py();

        let matched_words = RwLock::new(HashSet::with_capacity(haystack.len() / 16));
        py.allow_threads(|| {
            let matches: Vec<_> = ac_impl
                .try_find_iter(haystack.as_bytes())
                .expect("Aho-Corasick matching failed")
                .collect();

            let mut locked_matches = matched_words.write().unwrap();
            matches.iter().for_each(|m| {
                locked_matches.insert(&haystack[m.start()..m.end()]);
            });
        });

        let matched_words = matched_words.read().unwrap();

        // 在 GIL 下提取需要的数据
        let patterns = self_.patterns.clone();
        let components = self_.pattern_components.clone();
        let use_logic = self_.use_logic;

        let result_set = if use_logic {
            let result = RwLock::new(HashSet::with_capacity(patterns.len()));

            py.allow_threads(|| {
                components
                    .par_iter()
                    .enumerate()
                    .for_each(|(i, (pos_terms, neg_groups))| {
                        let all_positive = pos_terms
                            .par_iter()
                            .all(|term| matched_words.contains(term as &str));

                        let no_negative = !neg_groups.par_iter().any(|group| {
                            group
                                .par_iter()
                                .all(|term| matched_words.contains(term as &str))
                        });

                        if all_positive && no_negative {
                            result.write().unwrap().insert(patterns[i].clone());
                        }
                    });
            });
            result.into_inner().unwrap()
        } else {
            py.allow_threads(|| {
                patterns
                    .par_iter()
                    .filter(|pattern| matched_words.contains(pattern as &str))
                    .cloned()
                    .collect()
            })
        };

        Ok(PySet::new(py, result_set.iter().map(|s| PyString::new(py, s)))?.into())
    }
}

#[pymodule]
fn kwmatcher(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AhoMatcher>()?;
    Ok(())
}
