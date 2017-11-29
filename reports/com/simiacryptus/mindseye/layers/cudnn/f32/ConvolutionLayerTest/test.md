# ConvolutionLayer
## ConvolutionLayerTest
### Json Serialization
Code from [LayerTestBase.java:84](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L84) executed in 0.00 seconds: 
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.ConvolutionLayer",
      "id": "c88cbdf1-1c2a-4a5e-b964-890900000007",
      "isFrozen": false,
      "name": "ConvolutionLayer/c88cbdf1-1c2a-4a5e-b964-890900000007",
      "filter": {
        "dimensions": [
          1,
          1,
          4
        ],
        "data": [
          -0.664,
          -1.948,
          1.088,
          1.044
        ]
      },
      "strideX": 1,
      "strideY": 1
    }
```



### Example Input/Output Pair
Code from [LayerTestBase.java:121](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L121) executed in 0.00 seconds: 
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
    [[
    	[ [ -1.664, 0.652 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.8142720460891724, 3.9221601486206055 ] ]
    ]
```



### Reference Implementation
Code from [LayerTestBase.java:132](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L132) executed in 0.01 seconds: 
```java
    System.out.println(new GsonBuilder().setPrettyPrinting().create().toJson(referenceLayer.getJson()));
    getEquivalencyTester().test(referenceLayer, layer, inputPrototype);
```
Logging: 
```
    {
      "class": "com.simiacryptus.mindseye.layers.aparapi.ConvolutionLayer",
      "id": "c88cbdf1-1c2a-4a5e-b964-89090000000e",
      "isFrozen": false,
      "name": "ConvolutionLayer/c88cbdf1-1c2a-4a5e-b964-89090000000e",
      "filter": {
        "dimensions": [
          1,
          1,
          4
        ],
        "data": [
          -0.664,
          -1.948,
          1.088,
          1.044
        ]
      },
      "skip": {
        "dimensions": [
          1,
          1
        ]
      },
      "simple": true
    }
    Inputs: [
    	[ [ -1.664, 0.652 ] ]
    ]
    Error: [
    	[ [ 4.6089172256458255E-8, 1.486206056000583E-7 ] ]
    ]
    Accuracy:
    absoluteTol: 9.7355e-08 +- 5.1266e-08 [4.6089e-08 - 1.4862e-07] (2#)
    relativeTol: 1.5824e-08 +- 3.1222e-09 [1.2702e-08 - 1.8946e-08] (2#)
    
```

### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.02 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: ConvolutionLayer/c88cbdf1-1c2a-4a5e-b964-890900000007
    Inputs: [
    	[ [ -1.664, 0.652 ] ]
    ]
    output=[
    	[ [ 1.8142720460891724, 3.9221601486206055 ] ]
    ]
    measured/actual: [ [ -0.6639957427978516, -1.9478797912597656 ], [ 1.087188720703125, 1.0418891906738281 ] ]
    implemented/expected: [ [ -0.6639999747276306, -1.9479999542236328 ], [ 1.0880000591278076, 1.0440000295639038 ] ]
    error: [ [ 4.231929779052734E-6, 1.201629638671875E-4 ], [ -8.113384246826172E-4, -0.0021108388900756836 ] ]
    Component: ConvolutionLayer/c88cbdf1-1c2a-4a5e-b964-890900000007
    Inputs: [
    	[ [ -1.664, 0.652 ] ]
    ]
    Outputs: [
    	[ [ 1.8142720460891724, 3.9221601486206055 ] ]
    ]
    Measured Gradient: [ [ -1.6629695892333984, 0.0 ], [ 0.0, -1.6617774963378906 ], [ 0.6508827209472656, 0.0 ], [ 0.0, 0.6508827209472656 ] ]
    Implemented Gradient: [ [ -1.6640000343322754, 0.0 ], [ 0.0, -1.6640000343322754 ], [ 0.6520000100135803, 0.0 ], [ 0.0, 0.6520000100135803 ] ]
    Error: [ [ 0.0010304450988769531, 0.0 ], [ 0.0, 0.0022225379943847656 ], [ -0.0011172890663146973, 0.0 ], [ 0.0, -0.0011172890663146973 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 7.1118e-04 +- 7.9358e-04 [0.0000e+00 - 2.2225e-03] (12#)
    relativeTol: 5.1401e-04 +- 3.6466e-04 [3.1867e-06 - 1.0120e-03] (8#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.21 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 8.1770 +- 1.3979 [5.8221 - 15.1153]
    Learning performance: 5.8240 +- 0.7351 [5.0299 - 10.5556]
    
```

