use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PySet, PyString},
};
use std::collections::HashSet;

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
    fn new(use_logic: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let use_logic_value = match use_logic {
            Some(value) => value
                .extract::<bool>()
                .map_err(|_| PyValueError::new_err("use_logic must be a boolean"))?,
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
    fn build(&mut self, py: Python<'_>, patterns: &Bound<'_, PyAny>) -> PyResult<()> {
        let py_set = patterns
            .downcast::<PySet>()
            .map_err(|_| PyValueError::new_err("Patterns must be a set"))?;

        let mut valid_patterns = Vec::new();
        let mut original_patterns = Vec::new();
        let mut pattern_components = Vec::new();

        for pat in py_set.iter() {
            let py_string = pat
                .downcast::<PyString>()
                .map_err(|_| PyValueError::new_err("All patterns must be strings"))?;
            let pattern = py_string.to_string();
            if pattern.is_empty() {
                return Err(PyValueError::new_err("Pattern cannot be empty"));
            }
            original_patterns.push(pattern.clone());

            if self.use_logic {
                let mut segments = pattern.split('~');

                let positive_part = segments.next().unwrap_or("");
                let positive_terms: Vec<String> = positive_part
                    .split('&')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();

                if positive_terms.is_empty() {
                    return Err(PyValueError::new_err(
                        "Pattern must contain at least one positive term before '~'",
                    ));
                }

                let mut negative_term_groups: Vec<Vec<String>> = Vec::new();
                for segment in segments {
                    let terms: Vec<String> = segment
                        .split('&')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();

                    if !terms.is_empty() {
                        negative_term_groups.push(terms);
                    }
                }

                pattern_components.push((positive_terms.clone(), negative_term_groups.clone()));

                valid_patterns.extend(positive_terms);
                for group in &negative_term_groups {
                    valid_patterns.extend(group.clone());
                }
            } else {
                valid_patterns.push(pattern.clone());
                pattern_components.push((vec![pattern.clone()], vec![]));
            }
        }

        let ac_impl = py.allow_threads(|| {
            AhoCorasickBuilder::new()
                .match_kind(MatchKind::LeftmostLongest)
                .build(&valid_patterns)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;

        self.ac_impl = Some(ac_impl);
        self.patterns = original_patterns;
        self.pattern_components = pattern_components;

        Ok(())
    }

    #[pyo3(text_signature = "(haystack: str)")]
    fn find(self_: PyRef<'_, Self>, haystack: &Bound<'_, PyAny>) -> PyResult<Py<PySet>> {
        let haystack_str = haystack
            .downcast::<PyString>()
            .map_err(|_| PyValueError::new_err("haystack must be a string"))?
            .to_str()
            .map_err(|_| PyValueError::new_err("haystack contains invalid UTF-8"))?;

        let ac_impl = match &self_.ac_impl {
            Some(ac) => ac,
            None => {
                return Err(PyValueError::new_err(
                    "AhoCorasick not built. Call build() first.",
                ))
            }
        };

        let py = self_.py();

        let matches = ac_impl
            .try_find_iter(haystack_str.as_bytes())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let all_matches = py.allow_threads(|| matches.collect::<Vec<_>>());
        let mut matched_words: HashSet<String> = HashSet::new();

        for m in all_matches {
            let matched = &haystack_str[m.start()..m.end()];
            matched_words.insert(matched.to_string());
        }

        let mut result_set = HashSet::new();

        if self_.use_logic {
            for (i, (positive_terms, negative_term_groups)) in
                self_.pattern_components.iter().enumerate()
            {
                let all_positive_present = positive_terms
                    .iter()
                    .all(|term| matched_words.contains(term));

                if !all_positive_present {
                    continue;
                }

                let any_negative_group_present = negative_term_groups
                    .iter()
                    .any(|group| group.iter().all(|term| matched_words.contains(term)));

                if !any_negative_group_present {
                    result_set.insert(self_.patterns[i].clone());
                }
            }
        } else {
            for pattern in &self_.patterns {
                if matched_words.contains(pattern) {
                    result_set.insert(pattern.clone());
                }
            }
        }

        let result = PySet::new(py, result_set.iter().map(|s| PyString::new(py, s)))?;
        Ok(result.into())
    }
}

#[pymodule]
fn kwmatcher(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AhoMatcher>()?;
    Ok(())
}
