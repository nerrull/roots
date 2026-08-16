// Minimal cross-platform named-shared-memory mapping, used by SignalScope to hand
// captured audio to an external process (the scope_monitor app) without going
// through the Wwise SDK or any IPC library. Windows uses CreateFileMapping /
// MapViewOfFile; Mac/Linux use POSIX shm_open / mmap. Header-only so it can be
// pulled into both the plug-in build (premake/vcxproj) and the monitor app's
// CMake build with no extra project wiring.
#pragma once

#include <string>
#include <cstdint>
#include <cstddef>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#endif

namespace mi {

// Wraps one named shared-memory mapping. Not thread-safe to call Create/Open/Close
// concurrently on the same instance; the mapped memory itself is meant to be
// shared and is synchronized field-by-field with std::atomic in signal_scope_shm.h.
class ShmRegion
{
public:
    ShmRegion() = default;
    ~ShmRegion() { Close(); }

    ShmRegion(const ShmRegion&) = delete;
    ShmRegion& operator=(const ShmRegion&) = delete;

    // Creates the region if it doesn't already exist, otherwise opens the existing
    // one. Whichever process gets there first decides the size; everyone else must
    // request the same size (mismatches aren't detected here). Used by the writer
    // (the Wwise plug-in), which owns the buffer's lifetime.
    bool CreateOrOpen(const std::string& name, size_t sizeBytes)
    {
        Close();
        return Map(/*create=*/true, name, sizeBytes);
    }

    // Opens an existing region only; fails (returns false) if nothing by that name
    // is mapped yet. Used by read-only clients (the monitor app) so they don't
    // race a writer into creating an empty buffer of the wrong size.
    bool OpenExisting(const std::string& name, size_t sizeBytes)
    {
        Close();
        return Map(/*create=*/false, name, sizeBytes);
    }

    void Close()
    {
#if defined(_WIN32)
        if (m_data) { UnmapViewOfFile(m_data); m_data = nullptr; }
        if (m_handle) { CloseHandle(m_handle); m_handle = nullptr; }
#else
        if (m_data) { munmap(m_data, m_size); m_data = nullptr; }
        if (m_fd >= 0) { close(m_fd); m_fd = -1; }
#endif
        m_size = 0;
    }

    void* Data() const { return m_data; }
    size_t Size() const { return m_size; }
    bool IsValid() const { return m_data != nullptr; }

private:
    bool Map(bool create, const std::string& name, size_t sizeBytes)
    {
#if defined(_WIN32)
        const std::string mappedName = "Local\\" + name;
        if (create)
        {
            m_handle = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                (DWORD)((uint64_t)sizeBytes >> 32), (DWORD)(sizeBytes & 0xFFFFFFFFu), mappedName.c_str());
            if (!m_handle)
                return false;
        }
        else
        {
            m_handle = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, mappedName.c_str());
            if (!m_handle)
                return false;
        }

        m_data = MapViewOfFile(m_handle, FILE_MAP_ALL_ACCESS, 0, 0, sizeBytes);
        if (!m_data)
        {
            CloseHandle(m_handle);
            m_handle = nullptr;
            return false;
        }
        m_size = sizeBytes;
        return true;
#else
        const std::string posixName = "/" + name;
        const int flags = create ? (O_CREAT | O_RDWR) : O_RDWR;
        m_fd = shm_open(posixName.c_str(), flags, 0666);
        if (m_fd < 0)
            return false;

        if (create)
        {
            // ftruncate on an already-sized segment (a second creator racing the
            // first) is harmless: it just re-asserts the same size.
            if (ftruncate(m_fd, (off_t)sizeBytes) != 0)
            {
                close(m_fd);
                m_fd = -1;
                return false;
            }
        }

        m_data = mmap(nullptr, sizeBytes, PROT_READ | PROT_WRITE, MAP_SHARED, m_fd, 0);
        if (m_data == MAP_FAILED)
        {
            m_data = nullptr;
            close(m_fd);
            m_fd = -1;
            return false;
        }
        m_size = sizeBytes;
        return true;
#endif
    }

#if defined(_WIN32)
    HANDLE m_handle = nullptr;
#else
    int m_fd = -1;
#endif
    void* m_data = nullptr;
    size_t m_size = 0;
};

} // namespace mi
