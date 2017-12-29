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

package org.deeplearning4j.nn.multilayer;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;

public class MultiLayerNetwork extends Model {
  private org.deeplearning4j.nn.api.Layer[] layers;
  
  public MultiLayerNetwork(MultiLayerConfiguration multiLayerConfiguration) {
    
  }
  
  public org.deeplearning4j.nn.api.Layer[] getLayers() {
    return layers;
  }
  
  public void init() {
  }
}
