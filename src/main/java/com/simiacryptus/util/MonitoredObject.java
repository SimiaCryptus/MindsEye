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

package com.simiacryptus.util;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.function.Supplier;

/**
 * The type Monitored object.
 */
public class MonitoredObject implements MonitoredItem {
  
  private final Map<String, Object> items = new HashMap<>();
  
  /**
   * Add obj monitored object.
   *
   * @param key  the key
   * @param item the item
   * @return the monitored object
   */
  public MonitoredObject addObj(String key, MonitoredItem item) {
    items.put(key, item);
    return this;
  }
  
  /**
   * Add field monitored object.
   *
   * @param key  the key
   * @param item the item
   * @return the monitored object
   */
  public MonitoredObject addField(String key, Supplier<Object> item) {
    items.put(key, item);
    return this;
  }
  
  /**
   * Clear constants monitored object.
   *
   * @return the monitored object
   */
  public MonitoredObject clearConstants() {
    HashSet<String> keys = new HashSet<>(items.keySet());
    for (String k : keys) {
      Object v = items.get(k);
      if (v instanceof MonitoredObject) {
        ((MonitoredObject) v).clearConstants();
      }
      else if (!(v instanceof Supplier) && !(v instanceof MonitoredItem)) {
        items.remove(k);
      }
    }
    return this;
  }
  
  /**
   * Add const monitored object.
   *
   * @param key  the key
   * @param item the item
   * @return the monitored object
   */
  public MonitoredObject addConst(String key, Object item) {
    items.put(key, item);
    return this;
  }
  
  @Override
  public Map<String, Object> getMetrics() {
    HashMap<String, Object> returnValue = new HashMap<>();
    items.entrySet().stream().parallel().forEach(e -> {
      String k = e.getKey();
      Object v = e.getValue();
      if (v instanceof MonitoredItem) {
        returnValue.put(k, ((MonitoredItem) v).getMetrics());
      }
      else if (v instanceof Supplier) {
        returnValue.put(k, ((Supplier) v).get());
      }
      else {
        returnValue.put(k, v);
      }
    });
    return returnValue;
  }
}
