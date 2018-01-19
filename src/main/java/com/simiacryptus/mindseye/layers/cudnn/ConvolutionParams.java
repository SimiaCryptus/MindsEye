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

import com.simiacryptus.mindseye.lang.cudnn.Precision;

/**
 * The type Convolution params.
 */
public class ConvolutionParams {
  /**
   * The Input bands.
   */
  public final int inputBands;
  /**
   * The Output bands.
   */
  public final int outputBands;
  /**
   * The Precision.
   */
  public final Precision precision;
  /**
   * The Stride x.
   */
  public final int strideX;
  /**
   * The Stride y.
   */
  public final int strideY;
  /**
   * The Padding x.
   */
  public final Integer paddingX;
  /**
   * The Padding y.
   */
  public final Integer paddingY;
  /**
   * The Master filter dimensions.
   */
  public final int[] masterFilterDimensions;
  
  /**
   * Instantiates a new Convolution params.
   *
   * @param inputBands             the input bands
   * @param outputBands            the output bands
   * @param precision              the precision
   * @param strideX                the stride x
   * @param strideY                the stride y
   * @param paddingX               the padding x
   * @param paddingY               the padding y
   * @param masterFilterDimensions the master filter dimensions
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
