package com.simiacryptus.mindseye.net.dag;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/***
 * Builds a network NNLayer components, assumed to form a directed acyclic graph
 * with a single output. Supplied builder methods designed to build linear
 * sequence of units acting on the current output node.
 *
 * @author Andrew Charneski
 */
public abstract class DAGNetwork extends NNLayer<DAGNetwork> implements DAGNode {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(DAGNetwork.class);

  private static final long serialVersionUID = -5683282519002886564L;

  final LinkedHashMap<UUID, NNLayer<?>> byId = new LinkedHashMap<>();
  final LinkedHashMap<UUID, DAGNode> nodesById = new LinkedHashMap<>();
  public final LinkedHashMap<UUID, InputNode> inputNodes;
  public final List<UUID> inputHandles;

  public DAGNetwork(int inputs) {
    inputHandles = new ArrayList<>();
    inputNodes = new LinkedHashMap<>();
    for(int i=0;i<inputs;i++) {
      UUID key = UUID.randomUUID();
      inputHandles.add(key);
      inputNodes.put(key, new InputNode(this, key));
    }
  }

  public List<DAGNode> getNodes() {
    return Stream.concat(
            nodesById.values().stream(),
            getInput().stream()
    ).collect(Collectors.toList());
  }

  public final EvaluationContext singleExeCtx(final Tensor... input) {
    return buildExeCtx(NNResult.singleResultArray(input));
  }

  public EvaluationContext buildExeCtx(final NNResult... inputs) {
    assert(inputs.length == inputHandles.size());
    final EvaluationContext evaluationContext = new EvaluationContext();
    for (int i = 0; i < inputs.length; i++) {
      evaluationContext.cache.put(this.inputHandles.get(i), inputs[i]);
    }
    return evaluationContext;
  }

  protected EvaluationContext batchExeContext(final Tensor[][] batchData) {
    return this.buildExeCtx(NNResult.batchResultArray(batchData));
  }

  @Override
  public NNLayer<DAGNetwork> evolve() {
    if (0 == this.byId.values().stream().filter(l -> {
      final NNLayer<?> evolve = l.evolve();
      if (null != evolve && evolve != l)
        throw new RuntimeException("Not implemented: Substitution via evolution in DAGNetwork");
      return null != evolve;
    }).count())
      return null;
    else
      return this;
  }

  @Override
  public DAGNetwork freeze() {
    this.byId.values().forEach(l -> l.freeze());
    return super.freeze();
  }

  public NNLayer<?> get(final int i) {
    return this.byId.get(i);
  }

  @Override
  public NNLayer<?> getChild(final UUID id) {
    if (this.id.equals(id))
      return this;
    if (this.byId.containsKey(id))
      return this.byId.get(id);
    return this.byId.values().stream().map(x -> x.getChild(id)).findAny().orElse(null);
  }

  @Override
  public List<NNLayer<?>> getChildren() {
    return this.byId.values().stream().flatMap(l -> l.getChildren().stream()).distinct().sorted(Comparator.comparing(l -> l.getId())).collect(Collectors.toList());
  }

  public List<DAGNode> getInput() {
    ArrayList<DAGNode> list = new ArrayList<>();
    for(UUID key : inputHandles) list.add(inputNodes.get(key));
    return list;
  }

  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJson();
      JsonArray nodes = new JsonArray();
      nodesById.forEach((k,v)->nodes.add(v.toJson()));
      json.add("nodes", nodes);
    return json;
  }

  public NNLayer<?> getLayer(final DAGNode head) {
    if (head instanceof InnerNode)
      return DAGNetwork.this.byId.get(((InnerNode) head).id);
    else
      return null;
  }

  @Override
  public List<double[]> state() {
    return getChildren().stream().flatMap(l -> l.state().stream()).distinct().collect(Collectors.toList());
  }

    public abstract DAGNode getHead();

    @Override
    public NNResult get(EvaluationContext buildExeCtx) {
        return getHead().get(buildExeCtx);
    }

    @Override
    public NNResult eval(final NNResult... input) {
      return getHead().get(buildExeCtx(input));
    }

    public NNResult batch(final Tensor[][] data) {
      return getHead().get(batchExeContext(data));
    }

  public DAGNode add(final NNLayer<?> nextHead, final DAGNode... head) {
        this.byId.put(nextHead.getId(), nextHead);
        assert null != getInput();
        final InnerNode node = new InnerNode(this, nextHead, head);
        nodesById.put(nextHead.getId(), node);
        return node;
    }

  public DAGNode getInput(int index) {
    DAGNode input = inputNodes.get(inputHandles.get(index));
    assert null != input;
    return input;
  }

  @Override
  public NNLayer<?> getLayer() {
    return this;
  }

  @Override
  public JsonObject toJson() {
    final JsonObject json = new JsonObject();
    json.add("head", getHead().toJson());
    return json;
  }

}
