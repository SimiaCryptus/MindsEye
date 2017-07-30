package com.simiacryptus.mindseye.layers.cudnn;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.TensorArray;
import com.simiacryptus.mindseye.layers.TensorList;
import com.simiacryptus.util.ml.Tensor;

import java.util.Arrays;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public abstract class DirectCuDNNLayer extends NNLayer {

    public static class CuDNNTensorList implements TensorList {
        public final CuDNN.CuDNNPtr ptr;
        public final int length;
        public final int[] dimensions;

        public CuDNNTensorList(CuDNN.CuDNNPtr ptr, int length, int[] dimensions) {
            this.ptr = ptr;
            this.length = length;
            this.dimensions = dimensions;
        }

        private volatile TensorList _inner = null;
        public TensorList inner() {
            if(null == _inner) {
                synchronized (this) {
                    if(null == _inner) {
                        int outLength = Tensor.dim(dimensions);
                        final double[] outputBuffer = Tensor.obtain(outLength * length);
                        assert(0 < outputBuffer.length);
                        Tensor[] output = IntStream.range(0, length)
                                .mapToObj(dataIndex -> new Tensor(dimensions))
                                .toArray(i -> new Tensor[i]);
                        double[][] outputBuffers = Arrays.stream(output).map(x -> x.getData()).toArray(i -> new double[i][]);
                        assert(length == outputBuffers.length);
                        ptr.read(outputBuffer);
                        for (int i = 0; i< length; i++) {
                          assert outLength == outputBuffers[0 +i].length;
                          System.arraycopy(outputBuffer, i * outLength, outputBuffers[0 +i], 0, outLength);
                        }
                        assert Arrays.stream(output).flatMapToDouble(x-> Arrays.stream(x.getData())).allMatch(v->Double.isFinite(v));
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
            return inner().length();
        }

        @Override
        public Stream<Tensor> stream() {
            return inner().stream();
        }
    }

    public DirectCuDNNLayer() {
        super();
    }

    public DirectCuDNNLayer(JsonObject json) {
        super(json);
    }

    public static TensorList fromDevice(CuDNN.CuDNNPtr ptr, int length, int[] dimensions) {
      return new CuDNNTensorList(ptr, length, dimensions);
    }

    public static CuDNN.CuDNNPtr toDevice(TensorList data) {
        if(data instanceof CuDNNTensorList) {
            return ((CuDNNTensorList)data).ptr;
        } else {
          int listLength = data.length();
          int elementLength = data.get(0).dim();
          double[][] inputBuffers = data.stream().map(x -> x.getData()).toArray(i -> new double[i][]);
          final double[] inputBuffer = Tensor.obtain(elementLength * listLength);
          for (int i = 0; i< listLength; i++) {
            assert elementLength == inputBuffers[0 +i].length;
            System.arraycopy(inputBuffers[0 +i], 0, inputBuffer, i * elementLength, elementLength);
          }
          assert(0 < inputBuffer.length);
          CuDNN.CuDNNPtr ptr = CuDNN.write(inputBuffer);
          Tensor.recycle(inputBuffer);
          return ptr;
        }
    }

    public static Tensor fromDevice(CuDNN.CuDNNPtr filterData, int[] dimensions) {
      final Tensor weightGradient = new Tensor(dimensions);
      filterData.read(weightGradient.getData());
      return weightGradient;
    }
}
