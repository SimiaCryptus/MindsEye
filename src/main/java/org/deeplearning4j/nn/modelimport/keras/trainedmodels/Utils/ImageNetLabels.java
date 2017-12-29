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

package org.deeplearning4j.nn.modelimport.keras.trainedmodels.Utils;

import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Helper class with a static method that returns the label description
 *
 * @author susaneraly
 */
public class ImageNetLabels {
  
  //FIXME
  private static final String jsonUrl =
    "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json";
  private static ArrayList<String> predictionLabels = null;
  
  public static ArrayList<String> getLabels() {
    if (predictionLabels == null) {
      HashMap<String, ArrayList<String>> jsonMap;
      try {
        jsonMap = new ObjectMapper().readValue(new URL(jsonUrl), HashMap.class);
        predictionLabels = new ArrayList<>(jsonMap.size());
        for (int i = 0; i < jsonMap.size(); i++) {
          predictionLabels.add(jsonMap.get(String.valueOf(i)).get(1));
        }
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
    return predictionLabels;
  }
  
  /*
      Returns the description of the nth class in the 1000 classes of ImageNet
   */
  public static String getLabel(int n) {
    if (predictionLabels == null) {
      getLabels();
    }
    return predictionLabels.get(n);
  }
  
}
