package com.simiacryptus.mindseye;

import java.io.DataInputStream;
import java.io.IOException;
import java.util.Iterator;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

public final class BinaryChunkIterator implements Iterator<byte[]> {
  
  
  private DataInputStream in;
  private int recordSize;

  public BinaryChunkIterator(DataInputStream in, int recordSize){
    super();
    this.in = in;
    this.recordSize = recordSize;
  }
  
  @Override
  public boolean hasNext() {
    try {
      return 0<in.available();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
  
  @Override
  public byte[] next() {
    assert(hasNext());
    try {
      return read(in, recordSize);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public static <T> Stream<T> toStream(Iterator<T> iterator) {
    return toStream(iterator, 0);
  }

  public static <T> Stream<T> toStream(Iterator<T> iterator, int size) {
    return StreamSupport.stream(Spliterators.spliterator(iterator, size, Spliterator.ORDERED), false);
  }

  public static byte[] read(DataInputStream i, int s) throws IOException {
    byte[] b = new byte[s];
    int pos = 0;
    while(b.length > pos) {
      int read = i.read(b, pos, b.length-pos);
      if(0==read) {
        throw new RuntimeException();
      }
      pos += read;
    }
    return b;
  }

  public Stream<byte[]> toStream() {
    return toStream(this);
  }
}