#pragma once

#include <fstream>
#include <sstream>
#include <string>

#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/gzip.hpp>

namespace ORUtils
{
    /// Helper class for generating traces
    class Trace
    {
    public:
        /// Writes a compute instruction trace to the given stream
        static void writeComputeInstruction(std::ostream& os, const uint32_t workId, const uint32_t instId, const uint32_t cycles)
        {
            if (Trace::enabled_) {
                os << workId << ","
                   << instId << ","
                   << "compute" << ","
                   << cycles << ","
                   << "none" << ","
                   << "none" << ","
                   << 0x0 << "\n";
            }
        }
        
        /// Writes a memory instruction trace to the given stream
        template<class T>
        static void writeMemoryInstruction(std::ostream& os, const uint32_t workId, const uint32_t instId, const std::string& instType,
            const std::string& addrSpace, const std::string& addrType, const T* addr)
        {
            if (Trace::enabled_) {
                os << workId << ","
                   << instId << ","
                   << instType << ","
                   << 0 << ","
                   << addrSpace << ","
                   << addrType << ","
                   << addr << "\n";
            }
        }
    
        /// Writes trace to file
        static void writeTrace(const std::string& filename, const uint32_t frame, const std::stringstream& ss)
        {
            if (Trace::enabled_) {
                // Output file
                std::ofstream ofs{filename, std::ios_base::app | std::ios_base::binary};

                // Boost compressor
                boost::iostreams::filtering_streambuf<boost::iostreams::output> streambuf;
                streambuf.push(boost::iostreams::gzip_compressor());
                streambuf.push(ofs);

                // Writeable ostream from filtering_streambuf
                std::ostream os(&streambuf);

                // Write file
                os << "#" << frame << "\n";
                os << ss.str();

                // This is apparently necessary
                boost::iostreams::close(streambuf);
            }
        }

        /// Clears trace file
        static void clearFile(const std::string& filename)
        {
            std::ofstream ofs{filename, std::ios_base::out | std::ios_base::trunc};
            ofs.close();
        }

        // Whether tracing is enabled or not
        static bool enabled_;
    };
}
