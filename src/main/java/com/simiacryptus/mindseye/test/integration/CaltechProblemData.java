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

package com.simiacryptus.mindseye.test.integration;

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.data.Caltech101;
import com.simiacryptus.util.test.LabeledObject;

import javax.annotation.Nullable;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Caltech 101 Image Dataset.
 */
public class CaltechProblemData implements ImageProblemData {
  
  private final int imageSize;
  @Nullable
  private List<String> labels = null;
  
  /**
   * Instantiates a new Caltech problem data.
   */
  public CaltechProblemData() {
    this(256);
  }
  
  /**
   * Instantiates a new Caltech problem data.
   *
   * @param imageSize the image size
   */
  public CaltechProblemData(int imageSize) {
    this.imageSize = imageSize;
  }
  
  @Override
  public Stream<LabeledObject<Tensor>> trainingData() {
    return Caltech101.trainingDataStream().parallel().map(x -> x.map(y -> Tensor.fromRGB(TestUtil.resize(y.get(), getImageSize()))));
  }
  
  @Override
  public Stream<LabeledObject<Tensor>> validationData() {
    return trainingData();
  }
  
  /**
   * Gets image size.
   *
   * @return the image size
   */
  public int getImageSize() {
    return imageSize;
  }
  
  /**
   * Gets labels.
   *
   * @return the labels
   */
  @Nullable
  public List<String> getLabels() {
    if (null == labels) {
      synchronized (this) {
        if (null == labels) {
          labels = trainingData().map(x -> x.label).distinct().sorted().collect(Collectors.toList());
        }
      }
    }
    return labels;
  }
  
}
