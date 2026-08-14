#include "midi_in.h"

#import <CoreMIDI/CoreMIDI.h>
#import <Foundation/Foundation.h>

#include <mutex>
#include <set>

namespace midi {

namespace {

std::string EndpointName(MIDIEndpointRef ep) {
    CFStringRef s = nullptr;
    if (MIDIObjectGetStringProperty(ep, kMIDIPropertyDisplayName, &s) != noErr || !s)
        return "?";
    std::string out = [(__bridge NSString*)s UTF8String] ?: "?";
    CFRelease(s);
    return out;
}

}  // namespace

struct Input::Impl {
    MIDIClientRef client = 0;
    MIDIPortRef   port   = 0;
    std::set<MIDIEndpointRef> connected;
    std::vector<std::string>  names;

    std::mutex mu;
    std::vector<CC> queue;
    uint64_t count = 0;
    bool open = false;

    void push(int ch, int cc, int val) {
        std::lock_guard<std::mutex> lk(mu);
        // A controller swept fast enough can outrun a stalled frame; the cap
        // keeps a wedged render loop from growing this without bound. Dropping
        // the oldest is right for CC data -- only the newest value matters.
        if (queue.size() > 4096) queue.erase(queue.begin(), queue.begin() + 2048);
        queue.push_back(CC{ch, cc, val});
        ++count;
    }
};

namespace {

void ReadProc(const MIDIPacketList* pktlist, void* refCon, void*) {
    auto* impl = static_cast<Input::Impl*>(refCon);
    if (!impl || !pktlist) return;
    const MIDIPacket* p = &pktlist->packet[0];
    for (unsigned i = 0; i < pktlist->numPackets; ++i) {
        // Packets can carry several messages back to back. Only control change
        // (0xB0) is of interest; everything else is skipped by its own length,
        // because a running-status stream would otherwise desynchronise.
        for (UInt16 b = 0; b + 2 < p->length; ) {
            const uint8_t status = p->data[b];
            if ((status & 0xF0) == 0xB0) {
                impl->push(status & 0x0F, p->data[b + 1], p->data[b + 2]);
                b += 3;
            } else if ((status & 0xF0) == 0xC0 || (status & 0xF0) == 0xD0) {
                b += 2;
            } else if (status < 0x80) {
                break;              // running status: not decoded, bail out
            } else {
                b += 3;
            }
        }
        p = MIDIPacketNext(p);
    }
}

}  // namespace

Input::Input() : impl_(new Impl()) {}
Input::~Input() { close(); }

bool Input::open(std::string& err) {
    if (impl_->open) return true;
    OSStatus st = MIDIClientCreate(CFSTR("mirror_app"), nullptr, nullptr, &impl_->client);
    if (st != noErr) { err = "MIDIClientCreate failed"; return false; }
    st = MIDIInputPortCreate(impl_->client, CFSTR("in"), ReadProc, impl_.get(),
                             &impl_->port);
    if (st != noErr) {
        err = "MIDIInputPortCreate failed";
        MIDIClientDispose(impl_->client);
        impl_->client = 0;
        return false;
    }
    impl_->open = true;
    rescan();
    return true;
}

void Input::close() {
    if (!impl_ || !impl_->open) return;
    if (impl_->port) MIDIPortDispose(impl_->port);
    if (impl_->client) MIDIClientDispose(impl_->client);
    impl_->port = 0;
    impl_->client = 0;
    impl_->connected.clear();
    impl_->names.clear();
    impl_->open = false;
}

bool Input::isOpen() const { return impl_->open; }

void Input::rescan() {
    if (!impl_->open) return;
    const ItemCount n = MIDIGetNumberOfSources();
    for (ItemCount i = 0; i < n; ++i) {
        MIDIEndpointRef src = MIDIGetSource(i);
        if (!src || impl_->connected.count(src)) continue;
        if (MIDIPortConnectSource(impl_->port, src, nullptr) == noErr) {
            impl_->connected.insert(src);
            impl_->names.push_back(EndpointName(src));
        }
    }
}

int Input::sourceCount() const { return (int)impl_->connected.size(); }

std::string Input::deviceInfo() const {
    if (!impl_->open) return "closed";
    if (impl_->names.empty()) return "no devices";
    std::string s;
    for (size_t i = 0; i < impl_->names.size(); ++i) {
        if (i) s += ", ";
        s += impl_->names[i];
    }
    return s;
}

void Input::drain(std::vector<CC>& out) {
    out.clear();
    if (!impl_->open) return;
    std::lock_guard<std::mutex> lk(impl_->mu);
    out.swap(impl_->queue);
}

uint64_t Input::received() const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    return impl_->count;
}

}  // namespace midi
