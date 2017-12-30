/*
 * Copyright (c) 2017 by Andrew Charneski.
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

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.google.gson.JsonArray;
import com.google.gson.JsonPrimitive;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.stream.IntStream;

/**
 * The type Json util.
 */
public class JsonUtil {
  
  /**
   * Get double array double [ ].
   *
   * @param array the array
   * @return the double [ ]
   */
  public static double[] getDoubleArray(final JsonArray array) {
    return IntStream.range(0, array.size()).mapToDouble(i -> array.get(i).getAsDouble()).toArray();
  }
  
  /**
   * Get int array int [ ].
   *
   * @param array the array
   * @return the int [ ]
   */
  public static int[] getIntArray(final JsonArray array) {
    if (null == array) return null;
    return IntStream.range(0, array.size()).map(i -> array.get(i).getAsInt()).toArray();
  }
  
  /**
   * Gets json.
   *
   * @param kernelDims the kernel dims
   * @return the json
   */
  public static JsonArray getJson(final double[] kernelDims) {
    final JsonArray array = new JsonArray();
    for (final double k : kernelDims) {
      array.add(new JsonPrimitive(k));
    }
    return array;
  }
  
  /**
   * Gets json.
   *
   * @param kernelDims the kernel dims
   * @return the json
   */
  public static JsonArray getJson(final int[] kernelDims) {
    final JsonArray array = new JsonArray();
    for (final int k : kernelDims) {
      array.add(new JsonPrimitive(k));
    }
    return array;
  }
  
  /**
   * Write json.
   *
   * @param out the out
   * @param obj the obj
   * @throws IOException the io exception
   */
  public static void writeJson(final OutputStream out, final Object obj) throws IOException {
    final ObjectMapper mapper = new ObjectMapper().enableDefaultTyping(ObjectMapper.DefaultTyping.NON_FINAL)
                                                  .enable(SerializationFeature.INDENT_OUTPUT);
    final ByteArrayOutputStream buffer = new ByteArrayOutputStream();
    mapper.writeValue(buffer, obj);
    out.write(buffer.toByteArray());
  }
}
