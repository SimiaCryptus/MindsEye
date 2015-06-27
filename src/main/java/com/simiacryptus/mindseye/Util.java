package com.simiacryptus.mindseye;

import java.io.DataInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

public class Util {

  public static <T> Stream<T> toStream(Iterator<T> iterator) {
    return Util.toStream(iterator, 0);
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

  public static <T> List<T> shuffle(List<T> buffer, Random random) {
    ArrayList<T> list = new ArrayList<T>(buffer);
    Collections.shuffle(list);
    return list;
  }
  
}
