package com.simiacryptus.mindseye;

import java.io.DataInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.function.DoubleSupplier;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.Serializer;

import de.javakaffee.kryoserializers.EnumMapSerializer;
import de.javakaffee.kryoserializers.EnumSetSerializer;
import de.javakaffee.kryoserializers.KryoReflectionFactorySupport;

public class Util {
  
  public static void add(final DoubleSupplier f, final double[] data) {
    for (int i = 0; i < data.length; i++)
    {
      data[i] += f.getAsDouble();
    }
  }
  
  public static byte[] read(final DataInputStream i, final int s) throws IOException {
    final byte[] b = new byte[s];
    int pos = 0;
    while (b.length > pos) {
      final int read = i.read(b, pos, b.length - pos);
      if (0 == read) throw new RuntimeException();
      pos += read;
    }
    return b;
  }
  
  public static <T> List<T> shuffle(final List<T> buffer, final Random random) {
    final ArrayList<T> list = new ArrayList<T>(buffer);
    Collections.shuffle(list);
    return list;
  }
  
  public static <T> Stream<T> toStream(final Iterator<T> iterator) {
    return Util.toStream(iterator, 0);
  }
  
  public static <T> Stream<T> toStream(final Iterator<T> iterator, final int size) {
    return toStream(iterator, size, false);
  }
  
  public static <T> Stream<T> toStream(final Iterator<T> iterator, final int size, boolean parallel) {
    return StreamSupport.stream(Spliterators.spliterator(iterator, size, Spliterator.ORDERED), parallel);
  }
  
  private static final ThreadLocal<Kryo> threadKryo = new ThreadLocal<Kryo>(){
    
    @Override
    protected Kryo initialValue() {
      Kryo kryo = new KryoReflectionFactorySupport() {
        
        @Override
        public Serializer<?> getDefaultSerializer(@SuppressWarnings("rawtypes") final Class clazz) {
          if (EnumSet.class.isAssignableFrom(clazz)) {
            return new EnumSetSerializer();
          }
          if (EnumMap.class.isAssignableFrom(clazz)) {
            return new EnumMapSerializer();
          }
          return super.getDefaultSerializer(clazz);
        }
        
      };
      return kryo;
    }
    
  };
  public static Kryo kryo() {
    return threadKryo.get();
  }
  
}
