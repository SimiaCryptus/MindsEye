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

package com.simiacryptus.util.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * The type Percentile statistics.
 */
@SuppressWarnings("serial")
public class PercentileStatistics extends ScalarStatistics {
  
  private final List<double[]> values = new ArrayList<>();
  
  @Override
  public synchronized ScalarStatistics add(final double... values) {
    if (null != this.values) {
      this.values.add(Arrays.copyOf(values, values.length));
    }
    super.add(values);
    return null;
  }
  
  @Override
  public void clear() {
    values.clear();
    super.clear();
  }
  
  @Override
  public Map<String, Object> getMetrics() {
    final Map<String, Object> map = super.getMetrics();
    map.put("tp50", getPercentile(0.5));
    map.put("tp75", getPercentile(0.75));
    map.put("tp90", getPercentile(0.9));
    return map;
  }
  
  /**
   * Gets percentile.
   *
   * @param percentile the percentile
   * @return the percentile
   */
  public synchronized Double getPercentile(final double percentile) {
    if (null == values) return Double.NaN;
    return values.parallelStream().flatMapToDouble(x -> Arrays.stream(x)).sorted().skip((int) (percentile * values.size())).findFirst().orElse(Double.NaN);
  }
  
}
