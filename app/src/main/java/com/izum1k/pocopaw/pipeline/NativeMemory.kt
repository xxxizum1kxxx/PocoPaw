package com.izum1k.pocopaw.pipeline

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Direct buffers backed by `malloc` instead of `ByteBuffer.allocateDirect`. ART counts
 * allocateDirect against the 256 MB Java heap, which 16 MP RAW16 frames (32 MB each) exhaust after
 * eight; native memory only answers to the device's RAM. Every buffer has to be [free]d exactly once.
 */
object NativeMemory {
    init {
        System.loadLibrary("burstpipe")
    }

    fun allocate(bytes: Int): ByteBuffer = nativeAllocate(bytes.toLong())?.order(ByteOrder.nativeOrder())
        ?: throw OutOfMemoryError("native allocation of $bytes bytes failed")

    /** Copies [height] rows of [rowBytes] from [src] (strided by [srcRowStride]) into a pooled native buffer in one JNI call. */
    fun copyRows(src: ByteBuffer, srcRowStride: Int, rowBytes: Int, height: Int): ByteBuffer =
        nativeCopyRows(src, srcRowStride, rowBytes, height)?.order(ByteOrder.nativeOrder())
            ?: throw OutOfMemoryError("native row copy of ${rowBytes.toLong() * height} bytes failed")

    fun free(buffer: ByteBuffer) = nativeFree(buffer)

    @JvmStatic private external fun nativeAllocate(bytes: Long): ByteBuffer?

    @JvmStatic private external fun nativeCopyRows(src: ByteBuffer, srcRowStride: Int, rowBytes: Int, height: Int): ByteBuffer?

    @JvmStatic private external fun nativeFree(buffer: ByteBuffer)
}
