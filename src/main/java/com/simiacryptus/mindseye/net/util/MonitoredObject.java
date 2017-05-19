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

package com.simiacryptus.mindseye.net.util;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;

public class MonitoredObject implements MonitoredItem {
  
  private final Map<String, Object> items = new HashMap<>();
  
  public MonitoredObject addObj(String key, MonitoredItem item) {
    items.put(key, item);
    return this;
  }
  
  public MonitoredObject addField(String key, Supplier<Object> item) {
    items.put(key, item);
    return this;
  }
  
  public MonitoredObject addConst(String key, Object item) {
    items.put(key, item);
    return this;
  }
  
  @Override
  public Map<String, Object> getMetrics() {
    HashMap<String, Object> returnValue = new HashMap<>();
    items.forEach((k, v) -> {
      if (v instanceof MonitoredItem) {
        returnValue.put(k, ((MonitoredItem) v).getMetrics());
      } else if (v instanceof Supplier) {
        returnValue.put(k, ((Supplier) v).get());
      } else {
        returnValue.put(k, v);
      }
    });
    return returnValue;
  }
}
