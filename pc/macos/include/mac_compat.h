/* Force-included into the host units of the macOS build (build_macos.py), as
 * win_compat.h is on Windows: what Linux has and Darwin spells differently
 * or lacks. */
#pragma once
#include <time.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <mach/mach_time.h>
#ifndef MAP_FIXED_NOREPLACE
/* Darwin takes an address without MAP_FIXED as a loose hint. The guest
 * window (ROOD_GUEST_BASE, 8 GB up) is free at start-up: the executable sits
 * at 4 GB, the shared cache and the heaps far higher. */
#define MAP_FIXED_NOREPLACE MAP_FIXED
#endif
#ifndef MAP_32BIT
#define MAP_32BIT 0
#endif
#ifndef MAP_NORESERVE
#define MAP_NORESERVE 0
#endif
#ifndef SOCK_NONBLOCK
#define SOCK_NONBLOCK 0 /* the developer RPC socket blocks on macOS */
#endif
#ifndef SOCK_CLOEXEC
#define SOCK_CLOEXEC 0
#endif
#ifndef TIMER_ABSTIME
#define TIMER_ABSTIME 1
/* clock_nanosleep: the frame pacing sleeps to absolute CLOCK_MONOTONIC
 * deadlines. */
static inline int rood_clock_nanosleep(clockid_t clock, int flags, const struct timespec* request,
    struct timespec* remain)
{
    struct timespec now, wait;
    if (!(flags & TIMER_ABSTIME))
        return nanosleep(request, remain) ? errno : 0;
    clock_gettime(clock, &now);
    long long ns = (long long)(request->tv_sec - now.tv_sec) * 1000000000LL + (request->tv_nsec - now.tv_nsec);
    if (ns <= 0)
        return 0;
    wait.tv_sec = (time_t)(ns / 1000000000LL);
    wait.tv_nsec = (long)(ns % 1000000000LL);
    return nanosleep(&wait, NULL) ? errno : 0;
}
#define clock_nanosleep rood_clock_nanosleep
#endif
#ifndef SOCK_NONBLOCK_ACCEPT4
#define SOCK_NONBLOCK_ACCEPT4
/* accept4: Linux only; the flags are 0 here (SOCK_NONBLOCK, SOCK_CLOEXEC). */
static inline int accept4(int fd, struct sockaddr* address, socklen_t* length, int flags)
{
    (void)flags;
    return accept(fd, address, length);
}
#endif
