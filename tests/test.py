import pytest
from kwmatcher import AhoMatcher


def test_basic_matching():
    """测试基本匹配功能"""
    matcher = AhoMatcher(use_logic=True)
    matcher.build({"apple", "banana"})

    assert matcher.find("I like apple") == {"apple"}
    assert matcher.find("banana is good") == {"banana"}
    assert matcher.find("apple and banana") == {"apple", "banana"}
    assert matcher.find("no fruit") == set()


def test_chinese_matching():
    """测试中文匹配功能"""
    matcher = AhoMatcher(use_logic=True)
    matcher.build({"苹果", "香蕉"})

    assert matcher.find("我喜欢苹果") == {"苹果"}
    assert matcher.find("香蕉很好吃") == {"香蕉"}
    assert matcher.find("苹果和香蕉") == {"苹果", "香蕉"}
    assert matcher.find("没有水果") == set()


def test_combined_conditions():
    """测试组合条件匹配功能"""
    matcher = AhoMatcher(use_logic=True)
    matcher.build({"apple&banana~orange", "苹果&香蕉~橘子", "tea&coffee~milk", "a&b&c~d","A&B~C&D&E~F&G&H&I&J"})

    assert matcher.find("apple banana") == {"apple&banana~orange"}
    assert matcher.find("apple banana orange") == set()
    assert matcher.find("tea coffee") == {"tea&coffee~milk"}
    assert matcher.find("tea coffee milk") == set()

    assert matcher.find("苹果 香蕉") == {"苹果&香蕉~橘子"}
    assert matcher.find("苹果 香蕉 橘子") == set()

    assert matcher.find("a b c") == {"a&b&c~d"}
    assert matcher.find("a b c d") == set()
    assert matcher.find("A B C D E") == set()
    assert matcher.find("A B F G H I J") == set()
    assert matcher.find("A B C E F G H I") == {"A&B~C&D&E~F&G&H&I&J"}
    assert matcher.find("A B C D E F G H I J") == set()



def test_edge_cases():
    """测试临界条件"""
    with pytest.raises(TypeError):
        AhoMatcher({""})

    matcher = AhoMatcher(use_logic=True)
    
    with pytest.raises(TypeError):
        matcher.find(["123"])

    with pytest.raises(TypeError):
        matcher.build(["123"])
        
    with pytest.raises(ValueError):
        matcher.build({"~neg1 & neg2"})
        
    with pytest.raises(ValueError):
        matcher.build({"~"})
        
    with pytest.raises(ValueError):
        matcher.build({""})

    matcher.build({"test@123", "hello_world"})
    assert matcher.find("test@123 is valid") == {"test@123"}
    assert matcher.find("hello_world") == {"hello_world"}

    long_text = "a" * 10000 + "apple" + "b" * 10000
    assert matcher.find(long_text) == set()

    matcher.build({f"kw_{i:05d}" for i in range(10000)})

    assert matcher.find("kw_05000") == {"kw_05000"}
    assert matcher.find("prefix_kw_09999_suffix") == {"kw_09999"}
    assert matcher.find("kw_10000") == set()
    assert matcher.find("kw_abcd") == set()

    multiple_matches = matcher.find("kw_00000 kw_00123 kw_00999")
    assert multiple_matches == {"kw_00000", "kw_00123", "kw_00999"}

    long_test_case = (
        "start " + " ".join(f"kw_{i:05d}" for i in range(0, 10000, 100)) + " end"
    )
    expected_matches = {f"kw_{i:05d}" for i in range(0, 10000, 100)}
    assert matcher.find(long_test_case) == expected_matches


def test_case_sensitivity():
    """测试大小写敏感性"""
    patterns = {"Apple", "BANANA"}
    matcher = AhoMatcher(use_logic=True)
    matcher.build(patterns)

    assert matcher.find("apple") == set()
    assert matcher.find("APPLE") == set()
    assert matcher.find("Banana") == set()
    assert matcher.find("BANANA") == {"BANANA"}


def test_multi_separators():
    """测试多分隔符"""
    patterns = {"a~b&c", "x~y~z"}
    matcher = AhoMatcher(use_logic=True)
    matcher.build(patterns)

    assert matcher.find("a") == {"a~b&c"}
    assert matcher.find("a b") == {"a~b&c"}
    assert matcher.find("a b c") == set()
    assert matcher.find("x") == {"x~y~z"}
    assert matcher.find("x y") == set()
    assert matcher.find("x y z") == set()


def test_whitespace_handling():
    """测试空白字符处理"""
    patterns = {"  space  ", "tab\t", "new\nline"}
    matcher = AhoMatcher(use_logic=True)
    matcher.build(patterns)

    assert matcher.find("  space  ") == {"  space  "}
    assert matcher.find("tab\t") == {"tab\t"}
    assert matcher.find("new\nline") == {"new\nline"}


def test_mixed_language():
    """测试混合语言"""
    patterns = {"hello&你好", "foo&bar~baz"}
    matcher = AhoMatcher(use_logic=True)
    matcher.build(patterns)

    assert matcher.find("hello 你好") == {"hello&你好"}
    assert matcher.find("foo bar") == {"foo&bar~baz"}
    assert matcher.find("foo bar baz") == set()


def test_without_logic():
    """测试不使用逻辑匹配的情况"""
    patterns = {"hello&你好", "foo&bar~baz"}
    matcher = AhoMatcher(use_logic=False)
    matcher.build(patterns)

    assert matcher.find("hello 你好") == set()
    assert matcher.find("hello&你好") == {"hello&你好"}
    assert matcher.find("foo bar baz") == set()
    assert matcher.find("foo&bar~baz") == {"foo&bar~baz"}
