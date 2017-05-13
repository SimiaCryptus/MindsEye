package com.simiacryptus.mindseye.net.activation;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.net.DeltaSet;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

public class DropoutNoiseLayer extends NNLayer {

  private double value;
  public static final ThreadLocal<Random> random = new ThreadLocal<Random>(){
    @Override
    protected Random initialValue() {
      return new Random();
    }
  };

  public double getValue() {
    return value;
  }

  public DropoutNoiseLayer setValue(double value) {
    this.value = value;
    return this;
  }

  private final class Result extends NNResult {
    private final NNResult inObj;
    private final Tensor[] mask;

    private Result(final Tensor[] outputA, final NNResult inObj, Tensor[] mask) {
      super(outputA);
      this.inObj = inObj;
      this.mask = mask;
    }

    @Override
    public void accumulate(final DeltaSet buffer, final Tensor[] delta) {
      if (this.inObj.isAlive()) {
        Tensor[] passbackA = java.util.stream.IntStream.range(0, delta.length).mapToObj(dataIndex->{
          final double[] deltaData = delta[dataIndex].getData();
          final int[] dims = this.inObj.data[dataIndex].getDims();
          double[] maskData = mask[dataIndex].getData();
          final Tensor passback = new Tensor(dims);
          for (int i = 0; i < passback.dim(); i++) {
            passback.set(i, maskData[i] * deltaData[i]);
          }
          return passback;
        }).toArray(i->new Tensor[i]);
        this.inObj.accumulate(buffer, passbackA);
      }
    }

    @Override
    public boolean isAlive() {
      return this.inObj.isAlive() || !isFrozen();
    }

  }

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(DropoutNoiseLayer.class);

  private static final long serialVersionUID = -2105152439043901220L;
  long seed = random.get().nextLong();

  public DropoutNoiseLayer() {
    super();
    this.setValue(1.0);
  }

  public void shuffle() {
    seed = random.get().nextLong();
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    int itemCnt = inObj[0].data.length;
    Random random = new Random(seed);
    Tensor[] mask = IntStream.range(0, itemCnt).mapToObj(dataIndex->{
      final Tensor input = inObj[0].data[dataIndex];
      final Tensor output = input.map(x -> {
        return (random.nextDouble() < getValue())?0:1;
      });
      return output;
    }).toArray(i->new Tensor[i]);
    Tensor[] outputA = java.util.stream.IntStream.range(0, itemCnt).mapToObj(dataIndex->{
      final double[] input = inObj[0].data[dataIndex].getData();
      final double[] maskT = mask[dataIndex].getData();
      final Tensor output = new Tensor(inObj[0].data[dataIndex].getDims());
      double[] outputData = output.getData();
      for(int i=0;i<outputData.length;i++) {
        outputData[i] = input[i] * maskT[i];
      }
      return output;
    }).toArray(i->new Tensor[i]);
    return new Result(outputA, inObj[0], mask);
  }

  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJson();
    json.addProperty("value", this.getValue());
    return json;
  }

  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }

}
