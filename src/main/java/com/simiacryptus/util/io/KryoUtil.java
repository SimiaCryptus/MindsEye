/*
 * Copyright (c) 2018 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.util.io;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.Serializer;
import com.esotericsoftware.kryo.serializers.FieldSerializer;
import de.javakaffee.kryoserializers.EnumMapSerializer;
import de.javakaffee.kryoserializers.EnumSetSerializer;
import de.javakaffee.kryoserializers.KryoReflectionFactorySupport;

import java.util.EnumMap;
import java.util.EnumSet;

/**
 * The type Kryo util.
 */
public class KryoUtil {
  
  private static final ThreadLocal<Kryo> threadKryo = new ThreadLocal<Kryo>() {
    
    @Override
    protected Kryo initialValue() {
      final Kryo kryo = new KryoReflectionFactorySupport() {
        
        @Override
        public Serializer<?> getDefaultSerializer(@SuppressWarnings("rawtypes") final Class clazz) {
          if (EnumSet.class.isAssignableFrom(clazz)) {
            return new EnumSetSerializer();
          }
          if (EnumMap.class.isAssignableFrom(clazz)) {
            return new EnumMapSerializer();
          }
          final Serializer<?> serializer = super.getDefaultSerializer(clazz);
          if (serializer instanceof FieldSerializer) {
            ((FieldSerializer<?>) serializer).setCopyTransient(false);
          }
          return serializer;
        }
        
      };
      return kryo;
    }
    
  };
  
  /**
   * Kryo kryo.
   *
   * @return the kryo
   */
  public static Kryo kryo() {
    return KryoUtil.threadKryo.get();
  }
}
