namespace luisa::compute {
    class Device;
    class Stream;
    class CommandList;
}

namespace global {
    void init(const char* program_path);
    luisa::compute::Device& device();
    luisa::compute::Stream& stream();
    luisa::compute::CommandList& cmd_list();
}