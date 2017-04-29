package com.simiacryptus.mindseye.training;

import java.util.List;
import java.util.stream.IntStream;

import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.training.TrainingContext.TerminationCondition;
import com.simiacryptus.mindseye.core.delta.DeltaBuffer;
import com.simiacryptus.mindseye.core.delta.DeltaSet;
import com.simiacryptus.mindseye.core.delta.NNLayer.ConstNNResult;
import com.simiacryptus.mindseye.core.delta.NNResult;
import com.simiacryptus.mindseye.net.DAGNetwork;
import com.simiacryptus.mindseye.net.DAGNode;
import com.simiacryptus.mindseye.net.EvaluationContext;

public class GradientDescentTrainer implements RateTrainingComponent {

  public abstract static class RevertableStep {
    public double finalError;
    public double prevError;

    public RevertableStep(final double prevError, final double finalError) {
      super();
      this.prevError = prevError;
      this.finalError = finalError;
    }

    public abstract void revert();

  }

  private static final Logger log = LoggerFactory.getLogger(GradientDescentTrainer.class);

  public static DeltaSet collectVector(final NNResult nnResult) {
    final DeltaSet buffer = new DeltaSet();
    nnResult.accumulate(buffer);
    return buffer;
  }

  private double error = Double.POSITIVE_INFINITY;
  private long hash = Util.R.get().nextLong();
  private DAGNetwork net = null;
  private DAGNode primaryNode;
  private double rate = 0.1;
  private Tensor[][] trainingData = null;
  private int trainingSize = Integer.MAX_VALUE;
  private boolean verbose = false;

  public RevertableStep _step(final TrainingContext trainingContext) {
    final Tensor[][] data = selectTrainingData();
    if (data.length == 0)
      return new RevertableStep(Double.NaN, Double.NaN) {
        @Override
        public void revert() {
        }
      };
    final double prevError = validateMeanSum(trainingContext, data).rms;
    final List<DeltaBuffer> deltas = calcDelta(trainingContext, data).vector();
    deltas.stream().forEach(d -> d.write(this.rate));
    final double validationError = validateMeanSum(trainingContext, data).rms;
    return new RevertableStep(prevError, validationError) {
      @Override
      public void revert() {
        deltas.stream().forEach(d -> d.write(-GradientDescentTrainer.this.rate));
        validateMeanSum(trainingContext, data);
      }
    };
  }

  protected DeltaSet calcDelta(final TrainingContext trainingContext, final Tensor[][] data) {
    final EvaluationContext contexts = createBatchExeContext(trainingContext, data);
    return collectVector(getPrimaryNode(), contexts);
  }

  protected DeltaSet collectVector(final DAGNode primaryNode, final EvaluationContext contexts) {
    return collectVector(primaryNode.get(contexts)).scale(-getRate());
  }

  private NNResult eval(final TrainingContext trainingContext, final Tensor[][] trainingData, final DAGNode primaryNode) {
    final EvaluationContext collect = createBatchExeContext(trainingContext, trainingData);
    return primaryNode.get(collect);
  }

  private ValidationResults validateMeanSum(final TrainingContext trainingContext, final Tensor[][] validationSet) {
    assert 0 < validationSet.length;
    final NNResult eval = eval(trainingContext, validationSet, getPrimaryNode());
    final List<Tensor> evalData = java.util.Arrays.asList(eval.data);
    assert 0 < evalData.size();
    final double rms = evalData.stream().parallel().mapToDouble((Tensor x) -> x.sum()).average().getAsDouble();
    setError(rms);
    return new ValidationResults(evalData, rms);
  }

  @Override
  public Tensor[][] getTrainingData() {
    return this.trainingData;
  }

  @Override
  public synchronized double getError() {
    if (Double.isNaN(this.error))
      return validateMeanSum(new TrainingContext(), selectTrainingData()).rms;
    return this.error;
  }

  @Override
  public DAGNetwork getNet() {
    return this.net;
  }

  protected DAGNode getPrimaryNode() {
    return this.primaryNode;
  }

  @Override
  public double getRate() {
    return this.rate;
  }

  public Tensor[][] selectTrainingData() {
    final Tensor[][] rawData = getTrainingData();
    assert 0 < rawData.length;
    assert 0 < getTrainingSize();
    return java.util.Arrays.stream(rawData).parallel() //
        .sorted(java.util.Comparator.comparingLong(y -> System.identityHashCode(y) ^ this.hash)) //
        .limit(getTrainingSize()) //
        .toArray(i -> new Tensor[i][]);
  }

  public int getTrainingSize() {
    return this.trainingSize;
  }

  public EvaluationContext createBatchExeContext(final TrainingContext trainingContext, final Tensor[][] trainingData) {
    final DAGNetwork net = getNet();
    trainingContext.evaluations.increment();
    NNResult[] constNNResult = IntStream.range(0, trainingData[0].length).mapToObj(j->{
      Tensor[] array = IntStream.range(0, trainingData.length).mapToObj(i->trainingData[i][j]).toArray(i->new Tensor[i]);
      return new ConstNNResult(array);
    }).toArray(x->new NNResult[x]);
    return net.buildExeCtx(constNNResult);
  }

  public boolean isVerbose() {
    return this.verbose;
  }

  @Override
  public void reset() {
    setError(Double.NaN);
  }

  @Override
  public TrainingComponent setData(final Tensor[][] trainingData) {
    this.trainingData = trainingData;
    return this;
  }

  public TrainingComponent setError(final double error) {
    this.error = error;
    return this;
  }

  public TrainingComponent setNet(final DAGNetwork net) {
    this.net = net;
    setPrimaryNode(net.getHead());
    return this;
  }

  protected void setPrimaryNode(final DAGNode primaryNode) {
    this.primaryNode = primaryNode;
  }

  @Override
  public RateTrainingComponent setRate(final double dynamicRate) {
    assert Double.isFinite(dynamicRate);
    this.rate = dynamicRate;
    setError(Double.NaN);
    return this;
  }

  public GradientDescentTrainer setTrainingSize(final int trainingSize) {
    updateHash();
    this.trainingSize = trainingSize;
    return this;
  }

  public TrainingComponent setVerbose(final boolean verbose) {
    if (verbose) {
      this.verbose = true;
    }
    this.verbose = verbose;
    return this;
  }

  @Override
  public TrainingStep step(final TrainingContext trainingContext) throws TerminationCondition {
    final long startMs = System.currentTimeMillis();
    final RevertableStep result = _step(trainingContext);
    if (result.prevError == result.finalError) {
      if (this.verbose) {
        log.debug(String.format("Static: (%s)", result.prevError));
      }
      setError(result.finalError);
      trainingContext.gradientSteps.increment();
      return new TrainingStep(result.prevError, result.finalError, false);
    } else {
      setError(result.finalError);
      trainingContext.gradientSteps.increment();
      if (this.verbose) {
        log.debug(String.format("Step Complete in %.03f  - Error %s with rate %s and %s items - %s", //
            (System.currentTimeMillis() - startMs) / 1000., result.finalError, getRate(), Math.min(getTrainingSize(), trainingData.length), trainingContext));
      }
      return new TrainingStep(result.prevError, result.finalError, true);
    }
  }

  private void updateHash() {
    this.hash = Util.R.get().nextLong();
  }
}
