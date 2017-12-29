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

package org.nd4j.shade.jackson.databind;

import org.nd4j.shade.jackson.core.type.TypeReference;
import org.nd4j.shade.jackson.dataformat.yaml.YAMLFactory;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class ObjectMapper {
  public ObjectMapper(YAMLFactory yamlFactory) {
    throw new RuntimeException("NI");
  }
  
  public ObjectMapper() {
    throw new RuntimeException("NI");
  }
  
  public void enable(boolean failOnReadingDupTreeKey) {
    throw new RuntimeException("NI");
  }
  
  public void readTree(String s) throws IOException {
    throw new RuntimeException("NI");
  }
  
  public Map<String, Object> readValue(String json, TypeReference<HashMap<String, Object>> typeRef) {
    throw new RuntimeException("NI");
  }
  
  public String writeValueAsString(Map<String, Object> modelMapper) {
    return null;
  }
}
