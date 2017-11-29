# SimpleConvolutionLayer
## SimpleConvolutionLayerTest
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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f64.SimpleConvolutionLayer",
      "id": "c88cbdf1-1c2a-4a5e-b964-890900000f5b",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/c88cbdf1-1c2a-4a5e-b964-890900000f5b",
      "filter": {
        "dimensions": [
          1,
          1,
          1
        ],
        "data": [
          -1.184
        ]
      },
      "strideX": 1,
      "strideY": 1,
      "simple": false
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
    	[ [ -1.3 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ 1.5392 ] ]
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
      "id": "c88cbdf1-1c2a-4a5e-b964-890900000f5c",
      "isFrozen": false,
      "name": "ConvolutionLayer/c88cbdf1-1c2a-4a5e-b964-890900000f5c",
      "filter": {
        "dimensions": [
          1,
          1,
          1
        ],
        "data": [
          -1.184
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
    	[ [ -1.3 ] ]
    ]
    Error: [
    	[ [ 0.0 ] ]
    ]
    Accuracy:
    absoluteTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1#)
    relativeTol: 0.0000e+00 +- 0.0000e+00 [0.0000e+00 - 0.0000e+00] (1#)
    
```

### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.01 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: SimpleConvolutionLayer/c88cbdf1-1c2a-4a5e-b964-890900000f5b
    Inputs: [
    	[ [ -1.3 ] ]
    ]
    output=[
    	[ [ 1.5392 ] ]
    ]
    measured/actual: [ [ -1.1839999999985196 ] ]
    implemented/expected: [ [ -1.184 ] ]
    error: [ [ 1.4803713810351837E-12 ] ]
    Component: SimpleConvolutionLayer/c88cbdf1-1c2a-4a5e-b964-890900000f5b
    Inputs: [
    	[ [ -1.3 ] ]
    ]
    Outputs: [
    	[ [ 1.5392 ] ]
    ]
    Measured Gradient: [ [ -1.2999999999996348 ] ]
    Implemented Gradient: [ [ -1.3 ] ]
    Error: [ [ 3.652633751016765E-13 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 9.2282e-13 +- 5.5755e-13 [3.6526e-13 - 1.4804e-12] (2#)
    relativeTol: 3.8282e-13 +- 2.4234e-13 [1.4049e-13 - 6.2516e-13] (2#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.11 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 4.0327 +- 0.6770 [3.4511 - 8.5437]
    Learning performance: 2.8326 +- 0.2658 [2.5506 - 4.8617]
    
```

