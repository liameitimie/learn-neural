#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include "global.h"

using namespace luisa::compute;

namespace global {
    Context* ctx;
    Device _device;
    Stream _stream;
    CommandList _cmd_list;

    void init(const char* program_path) {
        ctx = new Context(program_path);
        _device = ctx->create_device("dx");
        _stream = _device.create_stream(StreamTag::GRAPHICS);
    }
    Device& device() {
        return _device;
    }
    Stream& stream() {
        return _stream;
    }
    CommandList& cmd_list() {
        return _cmd_list;
    }
}