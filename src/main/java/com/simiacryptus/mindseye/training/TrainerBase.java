package com.simiacryptus.mindseye.training;

import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;
import com.simiacryptus.mindseye.net.dag.DAGNetwork;
import com.simiacryptus.mindseye.net.dag.DAGNode;
import com.simiacryptus.mindseye.net.dag.EvaluationContext;
import com.simiacryptus.util.ml.Tensor;

import java.util.List;
import java.util.stream.IntStream;

public abstract class TrainerBase implements TrainingComponent {
    private Tensor[][] trainingData = null;
    private boolean verbose = false;
    private double error = Double.POSITIVE_INFINITY;
    private DAGNetwork net = null;
    private DAGNode primaryNode;

    @Override
    public Tensor[][] getTrainingData() {
      return this.trainingData;
    }

    public Tensor[][] selectTrainingData() {
        return getTrainingData();
    }

    @Override
    public synchronized double getError() {
        double error = this.error;
        if (Double.isNaN(error))
            return summarize(new TrainingContext(), selectTrainingData()).rms;
        return error;
    }

    protected NNResult eval(final TrainingContext trainingContext, final Tensor[][] data, final DAGNode primaryNode) {
        assert 0 < data.length;
        final EvaluationContext collect = createBatchExeContext(trainingContext, data);
        return primaryNode.get(collect);
    }

    protected NNResult eval(TrainingContext trainingContext, Tensor[][] data) {
        return eval(trainingContext, data, getPrimaryNode());
    }

    protected EvaluationContext createBatchExeContext(final TrainingContext trainingContext, final Tensor[][] batchData) {
        final DAGNetwork net = getNet();
        trainingContext.evaluations.increment();
        NNResult[] inputs = IntStream.range(0, batchData[0].length).mapToObj(inputIndex->{
            Tensor[] inputBatch = IntStream.range(0, batchData.length).mapToObj(trainingExampleId->
                    batchData[trainingExampleId][inputIndex]).toArray(i->new Tensor[i]);
            return new NNLayer.ConstNNResult(inputBatch);
        }).toArray(x->new NNResult[x]);
        return net.buildExeCtx(inputs);
    }

    protected ValidationResults summarize(final TrainingContext trainingContext, final Tensor[][] data) {
        return summarize(eval(trainingContext, data));
    }

    protected ValidationResults summarize(NNResult eval) {
        final List<Tensor> evalData = java.util.Arrays.asList(eval.data);
        assert 0 < evalData.size();
        final double meanSum = evalData.stream().parallel().mapToDouble((Tensor x) -> x.sum()).average().getAsDouble();
        setError(meanSum);
        return new ValidationResults(evalData, meanSum);
    }

    @Override
    public TrainingComponent setData(final Tensor[][] trainingData) {
        this.trainingData = trainingData;
        return this;
    }

    @Override
    public DAGNetwork getNet() {
      return this.net;
    }

    protected DAGNode getPrimaryNode() {
      return this.primaryNode;
    }

    public boolean isVerbose() {
      return this.verbose;
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

    public final TrainingComponent setVerbose(final boolean verbose) {
      if (verbose) {
        this.verbose = true;
      }
      this.verbose = verbose;
      return this;
    }
}
