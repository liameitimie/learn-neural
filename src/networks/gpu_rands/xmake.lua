target("gpu_rands")
    set_kind("headeronly")
    add_headerfiles("*.h")
    add_includedirs(".", {public=true})
    add_deps("lc-dsl")
target_end()