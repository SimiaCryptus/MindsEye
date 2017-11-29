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
      "id": "c88cbdf1-1c2a-4a5e-b964-890900000f6e",
      "isFrozen": false,
      "name": "FullyConnectedLayer/c88cbdf1-1c2a-4a5e-b964-890900000f6e",
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
          -0.23942742074080228,
          -0.44601584913199066,
          0.03195896070388802,
          -0.8736478362736179,
          0.06063193280523822,
          -0.7090145588600385,
          0.18159542692022976,
          -0.746595213169058,
          0.7918683614390686
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
    [[ 1.64, 0.364, 0.74 ]]
    --------------------
    Output: 
    [ -0.5762881664975426, -1.2618764267804607, 0.380313983594233 ]
```



### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.00 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: FullyConnectedLayer/c88cbdf1-1c2a-4a5e-b964-890900000f6e
    Inputs: [ 1.64, 0.364, 0.74 ]
    output=[ -0.5762881664975426, -1.2618764267804607, 0.380313983594233 ]
    measured/actual: [ [ -0.2394274207406788, -0.4460158491337296, 0.03195896070395943 ], [ -0.8736478362736388, 0.06063193280336776, -0.709014558859522 ], [ 0.18159542692020025, -0.7465952131679998, 0.7918683614394073 ] ]
    implemented/expected: [ [ -0.23942742074080228, -0.44601584913199066, 0.03195896070388802 ], [ -0.8736478362736179, 0.06063193280523822, -0.7090145588600385 ], [ 0.18159542692022976, -0.746595213169058, 0.7918683614390686 ] ]
    error: [ [ 1.2348455591393304E-13, -1.7389423234703827E-12, 7.140815716510929E-14 ], [ -2.098321516541546E-14, -1.8704621185250403E-12, 5.164757510556228E-13 ], [ -2.9504176879413535E-14, 1.0581535647702367E-12, 3.3872904481313526E-13 ] ]
    Component: FullyConnectedLayer/c88cbdf1-1c2a-4a5e-b964-890900000f6e
    Inputs: [ 1.64, 0.364, 0.74 ]
    Outputs: [ -0.5762881664975426, -1.2618764267804607, 0.380313983594233 ]
    Measured Gradient: [ [ 1.6399999999994197, 0.0, 0.0 ], [ 0.0, 1.6399999999983095, 0.0 ], [ 0.0, 0.0, 1.6399999999999748 ], [ 0.36400000000047505, 0.0, 0.0 ], [ 0.0, 0.36399999999936483, 0.0 ], [ 0.0, 0.0, 0.36400000000047505 ], [ 0.7399999999990747, 0.0, 0.0 ], [ 0.0, 0.7400000000012952, 0.0 ], [ 0.0, 0.0, 0.740000000000185 ] ]
    Implemented Gradient: [ [ 1.64, 0.0, 0.0 ], [ 0.0, 1.64, 0.0 ], [ 0.0, 0.0, 1.64 ], [ 0.364, 0.0, 0.0 ], [ 0.0, 0.364, 0.0 ], [ 0.0, 0.0, 0.364 ], [ 0.74, 0.0, 0.0 ], [ 0.0, 0.74, 0.0 ], [ 0.0, 0.0, 0.74 ] ]
    Error: [ [ -5.802025526691068E-13, 0.0, 0.0 ], [ 0.0, -1.6904255772942633E-12, 0.0 ], [ 0.0, 0.0, -2.5091040356528538E-14 ], [ 4.750644322371045E-13, 0.0, 0.0 ], [ 0.0, -6.351585923880521E-13, 0.0 ], [ 0.0, 0.0, 4.750644322371045E-13 ], [ -9.252598687226055E-13, 0.0, 0.0 ], [ 0.0, 1.2951861805277076E-12, 0.0 ], [ 0.0, 0.0, 1.8496315590255108E-13 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 3.3485e-13 +- 5.4330e-13 [0.0000e+00 - 1.8705e-12] (36#)
    relativeTol: 1.3684e-12 +- 3.4408e-12 [7.6497e-15 - 1.5425e-11] (18#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.01 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 0.2512 +- 0.0638 [0.1624 - 0.5785]
    Learning performance: 0.6472 +- 0.4049 [0.3220 - 2.5933]
    
```

