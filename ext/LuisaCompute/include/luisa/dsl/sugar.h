#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/string.h>

namespace luisa::compute::dsl_detail {
[[nodiscard]] LC_DSL_API luisa::string format_source_location(const char *file, int line) noexcept;
}// namespace luisa::compute::dsl_detail

#ifndef LUISA_COMPUTE_DESUGAR

#include <luisa/dsl/syntax.h>

#define $ ::luisa::compute::Var

#define $thread_id ::luisa::compute::thread_id()
#define $thread_x ::luisa::compute::thread_x()
#define $thread_y ::luisa::compute::thread_y()
#define $thread_z ::luisa::compute::thread_z()
#define $block_id ::luisa::compute::block_id()
#define $block_x ::luisa::compute::block_x()
#define $block_y ::luisa::compute::block_y()
#define $block_z ::luisa::compute::block_z()
#define $dispatch_id ::luisa::compute::dispatch_id()
#define $dispatch_x ::luisa::compute::dispatch_x()
#define $dispatch_y ::luisa::compute::dispatch_y()
#define $dispatch_z ::luisa::compute::dispatch_z()
#define $dispatch_size ::luisa::compute::dispatch_size()
#define $dispatch_size_x ::luisa::compute::dispatch_size_x()
#define $dispatch_size_y ::luisa::compute::dispatch_size_y()
#define $dispatch_size_z ::luisa::compute::dispatch_size_z()
#define $block_size ::luisa::compute::block_size()
#define $block_size_x ::luisa::compute::block_size_x()
#define $block_size_y ::luisa::compute::block_size_y()
#define $block_size_z ::luisa::compute::block_size_z()

#define $int $<int>
#define $uint $<::luisa::compute::uint>
#define $float $<float>
#define $bool $<bool>
#define $short $<short>
#define $ushort $<::luisa::compute::ushort>
#define $slong $<::luisa::compute::slong>
#define $ulong $<::luisa::compute::ulong>
#define $half $<::luisa::compute::half>

#define $int2 $<::luisa::compute::int2>
#define $uint2 $<::luisa::compute::uint2>
#define $float2 $<::luisa::compute::float2>
#define $bool2 $<::luisa::compute::bool2>
#define $short2 $<::luisa::compute::short2>
#define $ushort2 $<::luisa::compute::ushort2>
#define $slong2 $<::luisa::compute::slong2>
#define $ulong2 $<::luisa::compute::ulong2>
#define $half2 $<::luisa::compute::half2>

#define $int3 $<::luisa::compute::int3>
#define $uint3 $<::luisa::compute::uint3>
#define $float3 $<::luisa::compute::float3>
#define $bool3 $<::luisa::compute::bool3>
#define $short3 $<::luisa::compute::short3>
#define $ushort3 $<::luisa::compute::ushort3>
#define $slong3 $<::luisa::compute::slong3>
#define $ulong3 $<::luisa::compute::ulong3>
#define $half3 $<::luisa::compute::half3>

#define $int4 $<::luisa::compute::int4>
#define $uint4 $<::luisa::compute::uint4>
#define $float4 $<::luisa::compute::float4>
#define $bool4 $<::luisa::compute::bool4>
#define $short4 $<::luisa::compute::short4>
#define $ushort4 $<::luisa::compute::ushort4>
#define $slong4 $<::luisa::compute::slong4>
#define $ulong4 $<::luisa::compute::ulong4>
#define $half4 $<::luisa::compute::half4>

#define $float2x2 $<::luisa::compute::float2x2>
#define $float3x3 $<::luisa::compute::float3x3>
#define $float4x4 $<::luisa::compute::float4x4>

#define $array ::luisa::compute::ArrayVar
#define $constant ::luisa::compute::Constant
#define $shared ::luisa::compute::Shared
#define $buffer ::luisa::compute::BufferVar
#define $image ::luisa::compute::ImageVar
#define $volume ::luisa::compute::VolumeVar
#define $atomic ::luisa::compute::AtomicVar
#define $bindless ::luisa::compute::BindlessVar
#define $accel ::luisa::compute::AccelVar

#define $break ::luisa::compute::break_()
#define $continue ::luisa::compute::continue_()
#define $return(...) ::luisa::compute::return_(__VA_ARGS__)

#define $if(...)                                                                  \
    ::luisa::compute::detail::IfStmtBuilder::create_with_comment(                 \
        ::luisa::compute::dsl_detail::format_source_location(__FILE__, __LINE__), \
        __VA_ARGS__) %                                                            \
        [&]() noexcept
#define $else \
    / [&]() noexcept
#define $elif(...) \
    *([&] { return __VA_ARGS__; }) % [&]() noexcept

#define $loop                                                                       \
    ::luisa::compute::detail::LoopStmtBuilder::create_with_comment(                 \
        ::luisa::compute::dsl_detail::format_source_location(__FILE__, __LINE__)) % \
        [&]() noexcept
#define $while(...)                                                                 \
    ::luisa::compute::detail::LoopStmtBuilder::create_with_comment(                 \
        ::luisa::compute::dsl_detail::format_source_location(__FILE__, __LINE__)) / \
        [&]() noexcept {                                                            \
            $if (!(__VA_ARGS__)) { $break; };                                       \
        } %                                                                         \
        [&]() noexcept

#define $autodiff                                                                   \
    ::luisa::compute::detail::AutoDiffStmtBuilder::create_with_comment(             \
        ::luisa::compute::dsl_detail::format_source_location(__FILE__, __LINE__)) % \
        [&]() noexcept

#define $switch(...)                                                              \
    ::luisa::compute::detail::SwitchStmtBuilder::create_with_comment(             \
        ::luisa::compute::dsl_detail::format_source_location(__FILE__, __LINE__), \
        __VA_ARGS__) %                                                            \
        [&]() noexcept
#define $case(...)                                                                \
    ::luisa::compute::detail::SwitchCaseStmtBuilder::create_with_comment(         \
        ::luisa::compute::dsl_detail::format_source_location(__FILE__, __LINE__), \
        __VA_ARGS__) %                                                            \
        [&]() noexcept
#define $default                                                                    \
    ::luisa::compute::detail::SwitchDefaultStmtBuilder::create_with_comment(        \
        ::luisa::compute::dsl_detail::format_source_location(__FILE__, __LINE__)) % \
        [&]() noexcept

#define $for(x, ...)                                                                   \
    for (auto x : ::luisa::compute::dynamic_range_with_comment(                        \
             ::luisa::compute::dsl_detail::format_source_location(__FILE__, __LINE__), \
             __VA_ARGS__))                                                             \
    ::luisa::compute::detail::ForStmtBodyInvoke{} % [&]() noexcept

#define $comment(...) \
    ::luisa::compute::comment(__VA_ARGS__)
#define $comment_with_location(...)                                                                \
    $comment(luisa::string{__VA_ARGS__}                                                            \
                 .append(" [")                                                                     \
                 .append(::luisa::compute::dsl_detail::format_source_location(__FILE__, __LINE__)) \
                 .append("]"))

#endif
