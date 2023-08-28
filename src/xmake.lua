-- add_requires("fmt")
-- add_rules("lc_basic_settings")

target("test1")
    set_kind("binary")
    add_files("test1.cpp")
    add_deps("lc-runtime", "lc-dsl", "lc-vstl", "lc-backends-dummy")
    -- add_packages("fmt")

target("test2")
    set_kind("binary")
    add_files("test2.cpp")
    add_deps("lc-runtime", "lc-dsl", "lc-vstl", "lc-backends-dummy")

target("test3")
    set_kind("binary")
    add_files("test3.cpp")
    add_deps("lc-runtime", "lc-dsl", "lc-vstl", "lc-backends-dummy")

target("test4")
    set_kind("binary")
    add_files("test4.cpp")
    add_deps("lc-runtime", "lc-dsl", "lc-vstl", "lc-backends-dummy")

target("test5")
    set_kind("binary")
    add_files("test5.cpp")
    add_deps("lc-runtime", "lc-dsl", "lc-vstl", "lc-backends-dummy")

target("test6")
    set_kind("binary")
    add_files("test6.cpp")
    add_deps("lc-runtime", "lc-dsl", "lc-vstl", "lc-backends-dummy")

target("test7")
    set_kind("binary")
    add_files("test7.cpp")
    add_deps("lc-runtime", "lc-dsl", "lc-vstl", "lc-backends-dummy")

target("test8")
    set_kind("binary")
    add_files("test8.cpp")
    add_deps("lc-runtime", "lc-dsl", "lc-vstl", "lc-backends-dummy")

target("test9")
    set_kind("binary")
    add_files("test9.cpp")
    add_deps("lc-runtime", "lc-dsl", "lc-vstl", "lc-backends-dummy")

target("test10")
    set_kind("binary")
    add_files("test10.cpp")
    add_deps("lc-runtime", "lc-dsl", "lc-vstl", "lc-backends-dummy")

target("test11")
    set_kind("binary")
    add_files("test11.cpp")
    add_deps("lc-runtime", "lc-dsl", "lc-vstl", "lc-backends-dummy")

target("test12")
    set_kind("binary")
    add_files("test12.cpp")
    add_deps("lc-runtime", "lc-dsl", "lc-vstl", "lc-backends-dummy")

target("matmul_res1")
    set_kind("binary")
    add_files("matmul_res1.cpp")
    add_deps("lc-runtime", "lc-dsl", "lc-vstl", "lc-backends-dummy")