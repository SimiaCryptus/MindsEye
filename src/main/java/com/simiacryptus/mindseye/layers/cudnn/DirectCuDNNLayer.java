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
import com.simiacryptus.mindseye.layers.TensorArray;
import com.simiacryptus.mindseye.layers.TensorList;
import com.simiacryptus.util.ml.Tensor;
import jcuda.Sizeof;
import jcuda.jcudnn.cudnnTensorDescriptor;

import java.util.Arrays;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static jcuda.jcudnn.JCudnn.cudnnTransformTensor;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_FLOAT;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

/**
 * The type Direct cu dnn layer.
 */
public abstract class DirectCuDNNLayer extends NNLayer {
  
  /**
   * The type Cu dnn float tensor list.
   */
  public static class CuDNNFloatTensorList implements TensorList {
    /**
     * The Ptr.
     */
    public final CuDNN.CuDNNPtr ptr;
    /**
     * The Length.
     */
    public final int length;
    /**
     * The Dimensions.
     */
    public final int[] dimensions;
  
    /**
     * Instantiates a new Cu dnn float tensor list.
     *
     * @param ptr        the ptr
     * @param length     the length
     * @param dimensions the dimensions
     */
    public CuDNNFloatTensorList(CuDNN.CuDNNPtr ptr, int length, int[] dimensions) {
            this.ptr = ptr;
            this.length = length;
            this.dimensions = dimensions;
            assert(ptr.size == length * Tensor.dim(dimensions) * Sizeof.FLOAT);
            //assert this.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
        }

        private volatile TensorList _inner = null;
  
    /**
     * Inner tensor list.
     *
     * @return the tensor list
     */
    public TensorList inner() {
            if(null == _inner) {
                synchronized (this) {
                    if(null == _inner) {
                        int itemLength = Tensor.dim(dimensions);
                        final float[] buffer = new float[itemLength * length];
                        assert(0 < buffer.length);

                        //Arrays.stream(output).map(x -> x.getDataAsFloats()).toArray(i -> new float[i][]);
                        ptr.read(buffer);
                        //assert IntStream.range(0,buffer.length).mapToDouble(ii->buffer[ii]).allMatch(Double::isFinite);
                        float[][] floats = IntStream.range(0, length)
                                .mapToObj(dataIndex -> new float[itemLength])
                                .toArray(i -> new float[i][]);
                        for (int i = 0; i< length; i++) {
                            assert itemLength == floats[0 +i].length;
                            System.arraycopy(buffer, i * itemLength, floats[0 +i], 0, itemLength);
                        }
                        //assert Arrays.stream(output).flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
                        Tensor[] output = Arrays.stream(floats).map(floats2->{
                            return new Tensor(dimensions, floats2);
                        }).toArray(i->new Tensor[i]);
                        _inner = new TensorArray(output);
                    }
                }
            }
            return _inner;
        }

        @Override
        public Tensor get(int i) {
            return inner().get(i);
        }

        @Override
        public int length() {
            return length;
        }

        @Override
        public Stream<Tensor> stream() {
            return inner().stream();
        }
    }
  
  /**
   * The type Cu dnn double tensor list.
   */
  public static class CuDNNDoubleTensorList implements TensorList {
    /**
     * The Ptr.
     */
    public final CuDNN.CuDNNPtr ptr;
    /**
     * The Length.
     */
    public final int length;
    /**
     * The Dimensions.
     */
    public final int[] dimensions;
  
    /**
     * Instantiates a new Cu dnn double tensor list.
     *
     * @param ptr        the ptr
     * @param length     the length
     * @param dimensions the dimensions
     */
    public CuDNNDoubleTensorList(CuDNN.CuDNNPtr ptr, int length, int[] dimensions) {
            this.ptr = ptr;
            this.length = length;
            this.dimensions = dimensions;
            assert(ptr.size == length * Tensor.dim(dimensions) * Sizeof.DOUBLE);
            //assert this.stream().flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
        }

        private volatile TensorList _inner = null;
  
    /**
     * Inner tensor list.
     *
     * @return the tensor list
     */
    public TensorList inner() {
            if(null == _inner) {
                synchronized (this) {
                    if(null == _inner) {
                        int itemLength = Tensor.dim(dimensions);
                        final double[] outputBuffer = Tensor.obtain(itemLength * length);
                        assert(0 < outputBuffer.length);
                        Tensor[] output = IntStream.range(0, length)
                                .mapToObj(dataIndex -> new Tensor(dimensions))
                                .toArray(i -> new Tensor[i]);
                        double[][] outputBuffers = Arrays.stream(output).map(x -> x.getData()).toArray(i -> new double[i][]);
                        assert(length == outputBuffers.length);
                        ptr.read(outputBuffer);
                        for (int i = 0; i< length; i++) {
                          assert itemLength == outputBuffers[0 +i].length;
                          System.arraycopy(outputBuffer, i * itemLength, outputBuffers[0 +i], 0, itemLength);
                        }
                        //assert Arrays.stream(output).flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
                        Tensor.recycle(outputBuffer);
                        _inner = new TensorArray(output);
                    }
                }
            }
            return _inner;
        }

        @Override
        public Tensor get(int i) {
            return inner().get(i);
        }

        @Override
        public int length() {
            return length;
        }

        @Override
        public Stream<Tensor> stream() {
            return inner().stream();
        }
    }
  
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
  public static TensorList fromDeviceDouble(CuDNN.CuDNNPtr ptr, int length, int[] dimensions) {
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
  public static TensorList fromDeviceFloat(CuDNN.CuDNNPtr ptr, int length, int[] dimensions) {
        return new CuDNNFloatTensorList(ptr, length, dimensions);
    }
  
  /**
   * To device as double cu dnn . cu dnn ptr.
   *
   * @param data the data
   * @return the cu dnn . cu dnn ptr
   */
  public static CuDNN.CuDNNPtr toDeviceAsDouble(TensorList data) {
        if(data instanceof CuDNNDoubleTensorList) {
            return ((CuDNNDoubleTensorList)data).ptr;
//        } else if(data instanceof CuDNNFloatTensorList) {
//            CuDNNFloatTensorList floatData = (CuDNNFloatTensorList) data;
//            int[] dimensions = floatData.dimensions;
//            int length = floatData.length;
//            CuDNN.CuDNNResource<cudnnTensorDescriptor> fromFormat = CuDNN.newTensorDescriptor(
//                    CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, length, dimensions[2], dimensions[1], dimensions[0]);
//            CuDNN.CuDNNResource<cudnnTensorDescriptor> toFormat = CuDNN.newTensorDescriptor(
//                    CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, length, dimensions[2], dimensions[1], dimensions[0]);
//            CuDNN.CuDNNPtr destPtr = CuDNN.alloc(Sizeof.DOUBLE * length * Tensor.dim(dimensions[2], dimensions[1], dimensions[0]));
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
            CuDNN.CuDNNPtr ptr = CuDNN.write(inputBuffer);
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
  public static CuDNN.CuDNNPtr toDeviceAsFloat(TensorList data) {
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
            CuDNN.CuDNNPtr ptr = CuDNN.write(inputBuffer);
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
  public static Tensor fromDeviceFloat(CuDNN.CuDNNPtr filterData, int[] dimensions) {
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
  public static Tensor fromDeviceDouble(CuDNN.CuDNNPtr filterData, int[] dimensions) {
        final Tensor weightGradient = new Tensor(dimensions);
        filterData.read(weightGradient.getData());
        return weightGradient;
    }
}
