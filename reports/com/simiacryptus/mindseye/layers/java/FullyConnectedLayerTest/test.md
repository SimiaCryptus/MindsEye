# FullyConnectedLayer
## FullyConnectedLayerTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.00 seconds: 
```java
    JsonObject json = layer.getJson();
    NNLayer echo = NNLayer.fromJson(json);
    assert (echo != null) : "Failed to deserialize";
    assert (layer != echo) : "Serialization did not copy";
    Assert.assertEquals("Serialization not equal", layer, echo);
    return new GsonBuilder().setPrettyPrinting().create().toJson(json);
```

Returns: 

```
    {
      "class": "com.simiacryptus.mindseye.layers.java.FullyConnectedLayer",
      "id": "e2d0bffa-47dc-4875-864f-3d3d0000149f",
      "isFrozen": false,
      "name": "FullyConnectedLayer/e2d0bffa-47dc-4875-864f-3d3d0000149f",
      "outputDims": [
        3
      ],
      "inputDims": [
        3
      ],
      "weights": {
        "dimensions": [
          3,
          3
        ],
        "data": [
          0.2238041736627254,
          0.7137490290649375,
          0.19053922959861253,
          0.35151960227263584,
          -0.8209343185494066,
          -0.683844456664976,
          0.41181871178311846,
          -0.018280005712603783,
          -0.6152918320424763
        ]
      }
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
```java
    SimpleEval eval = SimpleEval.run(layer, inputPrototype);
    return String.format("--------------------\nInput: \n[%s]\n--------------------\nOutput: \n%s",
      Arrays.stream(inputPrototype).map(t->t.prettyPrint()).reduce((a,b)->a+",\n"+b).get(),
      eval.getOutput().prettyPrint());
```

Returns: 

```
    --------------------
    Input: 
    [[ -0.532, 1.696, -0.636 ]]
    --------------------
    Output: 
    [ 0.21519672437175708, -1.7603930040891242, -0.869841463471246 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Feedback for input 0
    Inputs: [ -0.532, 1.696, -0.636 ]
    Output: [ 0.21519672437175708, -1.7603930040891242, -0.869841463471246 ]
    Measured: [ [ 0.2238041736624119, 0.7137490290642923, 0.19053922959866298 ], [ 0.3515196022729761, -0.8209343185483498, -0.6838444566659163 ], [ 0.41181871178352925, -0.018280005713577907, -0.6152918320423417 ] ]
    Implemented: [ [ 0.2238041736627254, 0.7137490290649375, 0.19053922959861253 ], [ 0.35151960227263584, -0.8209343185494066, -0.683844456664976 ], [ 0.41181871178311846, -0.018280005712603783, -0.6152918320424763 ] ]
    Error: [ [ -3.134992265785286E-13, -6.45261621912141E-13, 5.0459636469213365E-14 ], [ 3.402278458963792E-13, 1.0568212971406865E-12, -9.403589018575076E-13 ], [ 4.107825191113079E-13, -9.741235595939202E-13, 1.3455903058456897E-13 ] ]
    Learning Gradient for weight set 0
    Inputs: [ -0.532, 1.696, -0.636 ]
    Outputs: [ 0.21519672437175708, -1.7603930040891242, -0.869841463471246 ]
    Measured Gradient: [ [ -0.53200000000031, 0.0, 0.0 ], [ 0.0, -0.5319999999997549, 0.0 ], [ 0.0, 0.0, -0.5319999999997549 ], [ 1.6959999999999198, 0.0, 0.0 ], [ 0.0, 1.6959999999999198, 0.0 ], [ 0.0, 0.0, 1.6959999999999198 ], [ -0.6359999999999699, 0.0, 0.0 ], [ 0.0, -0.636000000000525, 0.0 ], [ 0.0, 0.0, -0.6359999999994148 ] ]
    Implemented Gradient: [ [ -0.532, 0.0, 0.0 ], [ 0.0, -0.532, 0.0 ], [ 0.0, 0.0, -0.532 ], [ 1.696, 0.0, 0.0 ], [ 0.0, 1.696, 0.0 ], [ 0.0, 0.0, 1.696 ], [ -0.636, 0.0, 0.0 ], [ 0.0, -0.636, 0.0 ], [ 0.0, 0.0, -0.636 ] ]
    Error: [ [ -3.099742684753437E-13, 0.0, 0.0 ], [ 0.0, 2.4513724383723456E-13, 0.0 ], [ 0.0, 0.0, 2.4513724383723456E-13 ], [ -8.01581023779363E-14, 0.0, 0.0 ], [ 0.0, -8.01581023779363E-14, 0.0 ], [ 0.0, 0.0, -8.01581023779363E-14 ], [ 3.008704396734174E-14, 0.0, 0.0 ], [ 0.0, -5.250244683452365E-13, 0.0 ], [ 0.0, 0.0, 5.8519855627992E-13 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.9575e-13 +- 3.0023e-13 [0.0000e+00 - 1.0568e-12] (36#)
    relativeTol: 1.7818e-12 +- 6.0344e-12 [2.3632e-14 - 2.6645e-11] (18#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2253 +- 0.1037 [0.1795 - 1.1570]
    Learning performance: 0.5399 +- 0.3073 [0.3363 - 2.5363]
    
```

