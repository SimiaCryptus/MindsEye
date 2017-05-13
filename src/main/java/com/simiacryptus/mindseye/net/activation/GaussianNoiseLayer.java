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

public class GaussianNoiseLayer extends NNLayer {

  private double value;
  public static final ThreadLocal<Random> random = new ThreadLocal<Random>(){
    @Override
    protected Random initialValue() {
      return new Random();
    }
  };
  private long seed = random.get().nextLong();;

  public double getValue() {
    return value;
  }

  public GaussianNoiseLayer setValue(double value) {
    this.value = value;
    return this;
  }

  private final class Result extends NNResult {
    private final NNResult inObj;

    private Result(final Tensor[] outputA, final NNResult inObj) {
      super(outputA);
      this.inObj = inObj;
    }

    @Override
    public void accumulate(final DeltaSet buffer, final Tensor[] delta) {
      if (this.inObj.isAlive()) {
        Tensor[] passbackA = java.util.stream.IntStream.range(0, delta.length).mapToObj(dataIndex->{
          final double[] deltaData = delta[dataIndex].getData();
          final int[] dims = this.inObj.data[dataIndex].getDims();
          final Tensor passback = new Tensor(dims);
          for (int i = 0; i < passback.dim(); i++) {
            passback.set(i, deltaData[i]);
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
  private static final Logger log = LoggerFactory.getLogger(GaussianNoiseLayer.class);

  private static final long serialVersionUID = -2105152439043901220L;

  public GaussianNoiseLayer() {
    super();
    this.setValue(1.0);
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    int itemCnt = inObj[0].data.length;
    Random random = new Random(seed);
    Tensor[] outputA = java.util.stream.IntStream.range(0, itemCnt).mapToObj(dataIndex->{
      final Tensor input = inObj[0].data[dataIndex];
      final Tensor output = input.map(x -> {
        return x + random.nextGaussian() * getValue();
      });
      return output;
    }).toArray(i->new Tensor[i]);
    return new Result(outputA, inObj[0]);
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

  public void shuffle() {
    seed = random.get().nextLong();
  }
}
