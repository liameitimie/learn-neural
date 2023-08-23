-- add_requires("fmt")

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

target("matmul_res1")
    set_kind("binary")
    add_files("matmul_res1.cpp")
    add_deps("lc-runtime", "lc-dsl", "lc-vstl", "lc-backends-dummy")