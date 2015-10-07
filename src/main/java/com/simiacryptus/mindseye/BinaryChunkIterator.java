package com.simiacryptus.mindseye;

import java.io.DataInputStream;
import java.io.IOException;
import java.util.Iterator;
import java.util.stream.Stream;

public final class BinaryChunkIterator implements Iterator<byte[]> {

  private DataInputStream in;
  private int recordSize;

  public BinaryChunkIterator(final DataInputStream in, final int recordSize) {
    super();
    this.in = in;
    this.recordSize = recordSize;
  }

  @Override
  public boolean hasNext() {
    try {
      return 0 < this.in.available();
    } catch (final IOException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public byte[] next() {
    assert hasNext();
    try {
      return Util.read(this.in, this.recordSize);
    } catch (final IOException e) {
      throw new RuntimeException(e);
    }
  }

  public Stream<byte[]> toStream() {
    return Util.toStream(this);
  }
}
