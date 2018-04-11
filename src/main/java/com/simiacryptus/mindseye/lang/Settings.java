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

package com.simiacryptus.mindseye.lang;

import com.simiacryptus.mindseye.lang.cudnn.CudaSettings;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;

/**
 * The type Settings.
 */
public interface Settings {
  /**
   * The constant logger.
   */
  Logger logger = LoggerFactory.getLogger(CudaSettings.class);
  
  /**
   * Gets boolean.
   *
   * @param key          the key
   * @param defaultValue the default value
   * @return the boolean
   */
  static boolean get(final String key, final boolean defaultValue) {
    boolean value = Boolean.parseBoolean(System.getProperty(key, Boolean.toString(defaultValue)));
    logger.info(String.format("%s = %s", key, value));
    return value;
  }
  
  /**
   * Get t.
   *
   * @param <T>          the type parameter
   * @param key          the key
   * @param defaultValue the default value
   * @return the t
   */
  static <T extends Enum<T>> T get(final String key, @Nonnull final T defaultValue) {
    T value = Enum.valueOf((Class<T>) defaultValue.getClass().getSuperclass(), System.getProperty(key, defaultValue.toString().toUpperCase()));
    logger.info(String.format("%s = %s", key, value));
    return value;
  }
  
  /**
   * Gets int.
   *
   * @param key          the key
   * @param defaultValue the default value
   * @return the int
   */
  static String get(final String key, final String defaultValue) {
    String value = System.getProperty(key, defaultValue);
    logger.info(String.format("%s = %s", key, value));
    return value;
  }
  
  /**
   * Gets int.
   *
   * @param key          the key
   * @param defaultValue the default value
   * @return the int
   */
  static int get(final String key, final int defaultValue) {
    int value = Integer.parseInt(System.getProperty(key, Integer.toString(defaultValue)));
    logger.info(String.format("%s = %s", key, value));
    return value;
  }
  
  /**
   * Gets double.
   *
   * @param key          the key
   * @param defaultValue the default value
   * @return the double
   */
  static double get(final String key, final double defaultValue) {
    double value = Double.parseDouble(System.getProperty(key, Double.toString(defaultValue)));
    logger.info(String.format("%s = %s", key, value));
    return value;
  }
  
  /**
   * Gets long.
   *
   * @param key          the key
   * @param defaultValue the default value
   * @return the long
   */
  static long get(final String key, final long defaultValue) {
    long value = Long.parseLong(System.getProperty(key, Long.toString(defaultValue)));
    logger.info(String.format("%s = %s", key, value));
    return value;
  }
}
