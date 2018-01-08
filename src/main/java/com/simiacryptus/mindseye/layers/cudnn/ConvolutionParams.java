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

package com.simiacryptus.mindseye.layers.cudnn;

public class ConvolutionParams {
  public final int inputBands;
  public final int outputBands;
  public final Precision precision;
  public final int strideX;
  public final int strideY;
  public final Integer paddingX;
  public final Integer paddingY;
  public final int[] masterFilterDimensions;
  
  /**
   */
  public ConvolutionParams(int inputBands, int outputBands, Precision precision, int strideX, int strideY, Integer paddingX, Integer paddingY, int[] masterFilterDimensions) {
    this.inputBands = inputBands;
    this.outputBands = outputBands;
    this.precision = precision;
    this.strideX = strideX;
    this.strideY = strideY;
    this.paddingX = paddingX;
    this.paddingY = paddingY;
    this.masterFilterDimensions = masterFilterDimensions;
  }
  
}
