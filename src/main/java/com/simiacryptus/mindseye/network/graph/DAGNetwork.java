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

package com.simiacryptus.mindseye.network.graph;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.mindseye.layers.meta.WeightExtractor;
import com.simiacryptus.mindseye.layers.util.NNLayerWrapper;
import com.simiacryptus.util.MonitoredItem;
import com.simiacryptus.util.MonitoredObject;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.Map.Entry;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/***
 * Builds a network NNLayer components, assumed to form a directed acyclic graph
 * map a single output. Supplied builder methods designed to build linear
 * sequence of units acting on the current output node.
 *
 * @author Andrew Charneski
 */
public abstract class DAGNetwork extends NNLayer implements DAGNode {
  
  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJsonStub();
    JsonArray inputs = new JsonArray();
    json.add("inputs", inputs);
    inputHandles.forEach(uuid->inputs.add(new JsonPrimitive(uuid.toString())));
    JsonObject layerMap = new JsonObject();
    JsonObject nodeMap = new JsonObject();
    JsonObject links = new JsonObject();
    nodesById.forEach((nodeId, node) -> {
      JsonArray linkArray = new JsonArray();
      Arrays.stream(node.getInputs()).forEach(input->linkArray.add(new JsonPrimitive(input.getId().toString())));
      NNLayer layer = node.getLayer();
      nodeMap.addProperty(nodeId.toString(), layer.id.toString());
      layerMap.add(layer.id.toString(), layer.getJson());
  
      links.add(nodeId.toString(),linkArray);
    });
    json.add("nodes", nodeMap);
    json.add("layers", layerMap);
    json.add("links", links);
    JsonObject labels = new JsonObject();
    this.labels.forEach((k,v)->{
      labels.add(k.toString(),new JsonPrimitive(v.toString()));
    });
    json.add("labels", labels);
    json.addProperty("head", getHead().getId().toString());
    return json;
  }
  
  /**
   * Instantiates a new Dag network.
   *
   * @param json the json
   */
  protected DAGNetwork(JsonObject json) {
    super(json);
    inputHandles = new ArrayList<>();
    inputNodes = new LinkedHashMap<>();
    for(JsonElement item : json.getAsJsonArray("inputs")) {
      UUID key = UUID.fromString(item.getAsString());
      inputHandles.add(key);
      inputNodes.put(key, new InputNode(this, key));
    }
    JsonObject jsonNodes = json.getAsJsonObject("nodes");
    JsonObject jsonLayers = json.getAsJsonObject("layers");
    JsonObject jsonLinks = json.getAsJsonObject("links");
    JsonObject jsonLabels = json.getAsJsonObject("labels");
    Map<UUID, NNLayer> source_layersByNodeId = new HashMap<>();
    Map<UUID, NNLayer> source_layersByLayerId = new HashMap<>();
    for(Entry<String, JsonElement> e : jsonLayers.entrySet()) {
      source_layersByLayerId.put(UUID.fromString(e.getKey()), NNLayer.fromJson(e.getValue().getAsJsonObject()));
    }
    for(Entry<String, JsonElement> e : jsonNodes.entrySet()) {
      UUID nodeId = UUID.fromString(e.getKey());
      NNLayer layer = source_layersByLayerId.get(UUID.fromString(e.getValue().getAsString()));
      source_layersByNodeId.put(nodeId, layer);
    }
    for(Entry<String, JsonElement> e : jsonLabels.entrySet()) {
      this.labels.put(e.getKey(), UUID.fromString(e.getValue().getAsString()));
    }
    Map<UUID, List<UUID>> deserializedLinks = new HashMap<>();
    for(Entry<String, JsonElement> e : jsonLinks.entrySet()) {
      ArrayList<UUID> linkList = new ArrayList<>();
      for(JsonElement linkItem : e.getValue().getAsJsonArray()) {
        linkList.add(UUID.fromString(linkItem.getAsString()));
      }
      deserializedLinks.put(UUID.fromString(e.getKey()), linkList);
    }
    for(UUID key : this.labels.values()) {
      addAll(deserializedLinks, source_layersByNodeId, key);
    }
    UUID head = UUID.fromString(json.getAsJsonPrimitive("head").getAsString());
    addAll(deserializedLinks, source_layersByNodeId, head);
    for(NNLayer layer : source_layersByNodeId.values()) {
      if(layer instanceof WeightExtractor) {
        WeightExtractor weightExtractor = (WeightExtractor) layer;
        weightExtractor.setInner(source_layersByLayerId.get(weightExtractor.getInnerId()));
      }
    }
  }
  
  private void addAll(Map<UUID, List<UUID>> deserializedLinks, Map<UUID, NNLayer> deserializedNodes, UUID key) {
    if(layersById.containsKey(key)) return;
    if(inputNodes.containsKey(key)) return;
    NNLayer layer = deserializedNodes.get(key);
    if(layer == null) {
      throw new RuntimeException(String.format("%s is linked to but not defined", key));
    }
    List<UUID> links = deserializedLinks.get(key);
    if(null != links) for(UUID link : links) {
      addAll(deserializedLinks, deserializedNodes, link);
    }
    add(layer, getDependencies(deserializedLinks, key));
  }
  
  private DAGNode[] getDependencies(Map<UUID, List<UUID>> deserializedLinks, UUID e) {
    List<UUID> links = deserializedLinks.get(e);
    if(null == links) return new DAGNode[]{};
    return links.stream().map(id -> getNode(id)).toArray(i -> new DAGNode[i]);
  }
  
  private DAGNode getNode(UUID id) {
    DAGNode returnValue = nodesById.get(id);
    if(null == returnValue) {
      returnValue = inputNodes.get(id);
    }
    return returnValue;
  }
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(DAGNetwork.class);
  
  /**
   * The Input nodes.
   */
  public final LinkedHashMap<UUID, InputNode> inputNodes;
  /**
   * The Input handles.
   */
  public final List<UUID> inputHandles;
  /**
   * The Layers by id.
   */
  protected final LinkedHashMap<UUID, NNLayer> layersById = new LinkedHashMap<>();
  /**
   * The Nodes by id.
   */
  protected final LinkedHashMap<UUID, DAGNode> nodesById = new LinkedHashMap<>();
  
  /**
   * Labeled nodes
   */
  protected final LinkedHashMap<String, UUID> labels = new LinkedHashMap<>();
  
  /**
   * Gets by name.
   *
   * @param <T>  the type parameter
   * @param name the name
   * @return the by name
   */
  public <T extends NNLayer> T getByName(String name) {
    if(null == name) return null;
    AtomicReference<NNLayer> result = new AtomicReference<>();
    visit(n->{
      if(name.equals(n.getName())) result.set(n);
    });
    return (T) result.get();
  }
  
  /**
   * Visit.
   *
   * @param visitor the visitor
   */
  public void visit(Consumer<NNLayer> visitor) {
    layersById.values().forEach(layer->{
      if(layer instanceof DAGNetwork) {
        ((DAGNetwork)layer).visit(visitor);
      }
      if(layer instanceof NNLayerWrapper) {
        visitor.accept(((NNLayerWrapper)layer).getInner());
      }
      visitor.accept(layer);
    });
  }
  
  /**
   * Attach.
   *
   * @param obj the obj
   */
  public void attach(MonitoredObject obj) {
    visit(layer->{
      if(layer instanceof MonitoredItem) {
        obj.addObj(layer.getName(),(MonitoredItem)layer);
      }
    });
  }
  
  /**
   * Instantiates a new Dag network.
   *
   * @param inputs the inputs
   */
  public DAGNetwork(int inputs) {
    inputHandles = new ArrayList<>();
    inputNodes = new LinkedHashMap<>();
    for (int i = 0; i < inputs; i++) {
      addInput();
    }
  }
  
  public DAGNetwork addInput() {
    UUID key = UUID.randomUUID();
    inputHandles.add(key);
    inputNodes.put(key, new InputNode(this, key));
    return this;
  }
  
  public DAGNetwork removeLastInput() {
    int index = inputHandles.size() - 1;
    UUID key = inputHandles.remove(index);
    inputNodes.remove(key);
    return this;
  }
  
  /**
   * Gets nodes.
   *
   * @return the nodes
   */
  public List<DAGNode> getNodes() {
    return Stream.concat(
        nodesById.values().stream(),
        getInput().stream()
    ).collect(Collectors.toList());
  }
  
  /**
   * Single exe ctx evaluation context.
   *
   * @param input the input
   * @return the evaluation context
   */
  public final EvaluationContext singleExeCtx(final Tensor... input) {
    return buildExeCtx(NNResult.singleResultArray(input));
  }
  
  /**
   * Build exe ctx evaluation context.
   *
   * @param inputs the inputs
   * @return the evaluation context
   */
  public EvaluationContext buildExeCtx(final NNResult... inputs) {
    assert (inputs.length == inputHandles.size()) : inputs.length +" != "+ inputHandles.size();
    final EvaluationContext evaluationContext = new EvaluationContext();
    for (int i = 0; i < inputs.length; i++) {
      evaluationContext.cache.put(this.inputHandles.get(i), new CountingNNResult(inputs[i]));
    }
    return evaluationContext;
  }
  
  /**
   * Batch exe context evaluation context.
   *
   * @param batchData the batch data
   * @return the evaluation context
   */
  public EvaluationContext batchExeContext(final Tensor[][] batchData) {
    return this.buildExeCtx(NNResult.batchResultArray(batchData));
  }
  
  /**
   * Get nn layer.
   *
   * @param i the
   * @return the nn layer
   */
  public NNLayer get(final int i) {
    return this.layersById.get(i);
  }
  
  @Override
  public NNLayer getChild(final UUID id) {
    if (this.id.equals(id))
      return this;
    if (this.layersById.containsKey(id))
      return this.layersById.get(id);
    return this.layersById.values().stream().map(x -> x.getChild(id)).findAny().orElse(null);
  }
  
  @Override
  public List<NNLayer> getChildren() {
    return this.layersById.values().stream().flatMap(l -> l.getChildren().stream()).distinct().sorted(Comparator.comparing(l -> l.getId())).collect(Collectors.toList());
  }
  
  /**
   * Gets input.
   *
   * @return the input
   */
  public List<DAGNode> getInput() {
    ArrayList<DAGNode> list = new ArrayList<>();
    for (UUID key : inputHandles) list.add(inputNodes.get(key));
    return list;
  }
  
  /**
   * Gets layer.
   *
   * @param head the head
   * @return the layer
   */
  public NNLayer getLayer(final DAGNode head) {
    if (head instanceof InnerNode)
      return DAGNetwork.this.layersById.get(((InnerNode) head).id);
    else
      return null;
  }
  
  @Override
  public List<double[]> state() {
    return getChildren().stream().flatMap(l -> l.state().stream()).distinct().collect(Collectors.toList());
  }
  
  /**
   * Gets head.
   *
   * @return the head
   */
  public abstract DAGNode getHead();
  
  /**
   * Gets a labeled node
   * @param key
   * @return
   */
  public DAGNode getByLabel(String key) {
    return nodesById.get(labels.get(key));
  }
  
  @Override
  public NNResult get(NNExecutionContext nncontext, EvaluationContext buildExeCtx) {
    return getHead().get(nncontext, buildExeCtx);
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult[] input) {
    return getHead().get(nncontext, buildExeCtx(input));
  }
  
  @Override
  public DAGNetwork freeze() {
    visit(new Consumer<NNLayer>() {
      @Override
      public void accept(NNLayer layer) {
        layer.freeze();
      }
    });
    return this;
  }
  
  /**
   * Add dag node.
   *
   * @param nextHead the next head
   * @param head     the head
   * @return the dag node
   */
  public DAGNode add(final NNLayer nextHead, final DAGNode... head) {
    return add(null, nextHead, head);
  }
  
  /**
   * Add dag node.
   *
   *
   * @param label
   * @param nextHead the next head
   * @param head     the head
   * @return the dag node
   */
  public DAGNode add(String label, final NNLayer nextHead, final DAGNode... head) {
    assert null != getInput();
    final InnerNode node = new InnerNode(this, nextHead, head);
    this.layersById.put(nextHead.getId(), nextHead);
    nodesById.put(nextHead.getId(), node);
    if(null != label) labels.put(label,nextHead.getId());
    return node;
  }
  
  /**
   * Gets input.
   *
   * @param index the index
   * @return the input
   */
  public DAGNode getInput(int index) {
    DAGNode input = inputNodes.get(inputHandles.get(index));
    assert null != input;
    return input;
  }
  
  @Override
  public NNLayer getLayer() {
    return this;
  }
  
}
