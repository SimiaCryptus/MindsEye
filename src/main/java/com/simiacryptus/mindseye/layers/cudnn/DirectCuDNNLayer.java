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

package com.simiacryptus.mindseye.layers.cudnn;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.TensorList;
import com.simiacryptus.util.ml.Tensor;

/**
 * The type Direct cu dnn layer.
 */
public abstract class DirectCuDNNLayer extends NNLayer {
  
  /**
   * Instantiates a new Direct cu dnn layer.
   */
  public DirectCuDNNLayer() {
        super();
    }
  
  /**
   * Instantiates a new Direct cu dnn layer.
   *
   * @param json the json
   */
  public DirectCuDNNLayer(JsonObject json) {
        super(json);
    }
  
  /**
   * From device double tensor list.
   *
   * @param ptr        the ptr
   * @param length     the length
   * @param dimensions the dimensions
   * @return the tensor list
   */
  public static TensorList fromDeviceDouble(CudaPtr ptr, int length, int[] dimensions) {
        return new CuDNNDoubleTensorList(ptr, length, dimensions);
    }
  
  /**
   * From device float tensor list.
   *
   * @param ptr        the ptr
   * @param length     the length
   * @param dimensions the dimensions
   * @return the tensor list
   */
  public static TensorList fromDeviceFloat(CudaPtr ptr, int length, int[] dimensions) {
        return new CuDNNFloatTensorList(ptr, length, dimensions);
    }
  
  /**
   * To device as double cu dnn . cu dnn ptr.
   *
   * @param data the data
   * @return the cu dnn . cu dnn ptr
   */
  public static CudaPtr toDeviceAsDouble(int deviceId, TensorList data) {
        if(data instanceof CuDNNDoubleTensorList) {
            return ((CuDNNDoubleTensorList)data).ptr;
//        } else if(data instanceof CuDNNFloatTensorList) {
//            CuDNNFloatTensorList floatData = (CuDNNFloatTensorList) data;
//            int[] dimensions = floatData.dimensions;
//            int length = floatData.length;
//            CuDNN.CudaResource<cudnnTensorDescriptor> fromFormat = CuDNN.newTensorDescriptor(
//                    CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, length, dimensions[2], dimensions[1], dimensions[0]);
//            CuDNN.CudaResource<cudnnTensorDescriptor> toFormat = CuDNN.newTensorDescriptor(
//                    CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, length, dimensions[2], dimensions[1], dimensions[0]);
//            CuDNN.CudaPtr destPtr = CuDNN.alloc(Sizeof.DOUBLE * length * Tensor.dim(dimensions[2], dimensions[1], dimensions[0]));
//            CuDNN.devicePool.with(cudnn->{
//                cudnnTransformTensor(cudnn, );
//            });
//            return destPtr;
        } else {
            int listLength = data.length();
            int elementLength = data.get(0).dim();
            double[][] inputBuffers = data.stream().map(x -> x.getData()).toArray(i -> new double[i][]);
            final double[] inputBuffer = Tensor.obtain(elementLength * listLength);
            for (int i = 0; i< listLength; i++) {
                assert elementLength == inputBuffers[0 +i].length;
                System.arraycopy(inputBuffers[0 +i], 0, inputBuffer, i * elementLength, elementLength);
            }
            //assert(0 < inputBuffer.length);
            CudaPtr ptr = CuDNN.write(deviceId, inputBuffer);
            Tensor.recycle(inputBuffer);
            return ptr;
        }
    }
  
  /**
   * To device as float cu dnn . cu dnn ptr.
   *
   * @param data the data
   * @return the cu dnn . cu dnn ptr
   */
  public static CudaPtr toDeviceAsFloat(int deviceId, TensorList data) {
        if(data instanceof CuDNNFloatTensorList) {
            return ((CuDNNFloatTensorList)data).ptr;
//        } else if(data instanceof CuDNNDoubleTensorList) {
//            return ((CuDNNDoubleTensorList)data).ptr;
        } else {
            int listLength = data.length();
            int elementLength = data.get(0).dim();
            float[][] inputBuffers = data.stream().map(x -> x.getDataAsFloats()).toArray(i -> new float[i][]);
            final float[] inputBuffer = new float[elementLength * listLength];
            for (int i = 0; i< listLength; i++) {
                assert elementLength == inputBuffers[0 +i].length;
                System.arraycopy(inputBuffers[0 +i], 0, inputBuffer, i * elementLength, elementLength);
            }
            assert(0 < inputBuffer.length);
            //assert isNontrivial(inputBuffer);
            CudaPtr ptr = CuDNN.write(deviceId, inputBuffer);
            return ptr;
        }
    }
  
  /**
   * Is nontrivial boolean.
   *
   * @param data the data
   * @return the boolean
   */
  public static boolean isNontrivial(float[] data) {
        for(int i=0;i<data.length;i++) if(!Double.isFinite(data[i])) return false;
        for(int i=0;i<data.length;i++) if(data[i] != 0) return true;
        return false;
    }
  
  /**
   * Is nontrivial boolean.
   *
   * @param data the data
   * @return the boolean
   */
  public static boolean isNontrivial(double[] data) {
        for(int i=0;i<data.length;i++) if(!Double.isFinite(data[i])) return false;
        for(int i=0;i<data.length;i++) if(data[i] != 0) return true;
        return false;
    }
  
  /**
   * From device float tensor.
   *
   * @param filterData the filter data
   * @param dimensions the dimensions
   * @return the tensor
   */
  public static Tensor fromDeviceFloat(CudaPtr filterData, int[] dimensions) {
        final Tensor weightGradient = new Tensor(dimensions);
        int length = weightGradient.dim();
        float[] data = new float[length];
        filterData.read(data);
        double[] doubles = weightGradient.getData();
        for(int i = 0; i< length; i++) doubles[i] = data[i];
        return weightGradient;
    }
  
  /**
   * From device double tensor.
   *
   * @param filterData the filter data
   * @param dimensions the dimensions
   * @return the tensor
   */
  public static Tensor fromDeviceDouble(CudaPtr filterData, int[] dimensions) {
        final Tensor weightGradient = new Tensor(dimensions);
        filterData.read(weightGradient.getData());
        return weightGradient;
    }
}
