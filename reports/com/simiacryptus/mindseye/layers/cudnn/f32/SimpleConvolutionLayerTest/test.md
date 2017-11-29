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
      "class": "com.simiacryptus.mindseye.layers.cudnn.f32.SimpleConvolutionLayer",
      "id": "c88cbdf1-1c2a-4a5e-b964-8909000002f8",
      "isFrozen": false,
      "name": "SimpleConvolutionLayer/c88cbdf1-1c2a-4a5e-b964-8909000002f8",
      "filter": {
        "dimensions": [
          1,
          1,
          1
        ],
        "data": [
          -1.72
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
    	[ [ 1.164 ] ]
    ]]
    --------------------
    Output: 
    [
    	[ [ -2.002080202102661 ] ]
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
      "id": "c88cbdf1-1c2a-4a5e-b964-8909000002f9",
      "isFrozen": false,
      "name": "ConvolutionLayer/c88cbdf1-1c2a-4a5e-b964-8909000002f9",
      "filter": {
        "dimensions": [
          1,
          1,
          1
        ],
        "data": [
          -1.72
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
    	[ [ 1.164 ] ]
    ]
    Error: [
    	[ [ -2.0210266127307364E-7 ] ]
    ]
    Accuracy:
    absoluteTol: 2.0210e-07 +- 0.0000e+00 [2.0210e-07 - 2.0210e-07] (1#)
    relativeTol: 5.0473e-08 +- 0.0000e+00 [5.0473e-08 - 5.0473e-08] (1#)
    
```

### Differential Validation
Code from [LayerTestBase.java:139](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L139) executed in 0.01 seconds: 
```java
    getDerivativeTester().test(layer, inputPrototype);
```
Logging: 
```
    Component: SimpleConvolutionLayer/c88cbdf1-1c2a-4a5e-b964-8909000002f8
    Inputs: [
    	[ [ 1.164 ] ]
    ]
    output=[
    	[ [ -2.002080202102661 ] ]
    ]
    measured/actual: [ [ -1.7189979553222656 ] ]
    implemented/expected: [ [ -1.7200000286102295 ] ]
    error: [ [ 0.0010020732879638672 ] ]
    Component: SimpleConvolutionLayer/c88cbdf1-1c2a-4a5e-b964-8909000002f8
    Inputs: [
    	[ [ 1.164 ] ]
    ]
    Outputs: [
    	[ [ -2.002080202102661 ] ]
    ]
    Measured Gradient: [ [ 1.1658668518066406 ] ]
    Implemented Gradient: [ [ 1.1640000343322754 ] ]
    Error: [ [ 0.0018668174743652344 ] ]
    Finite-Difference Derivative Accuracy:
    absoluteTol: 1.4344e-03 +- 4.3237e-04 [1.0021e-03 - 1.8668e-03] (2#)
    relativeTol: 5.4632e-04 +- 2.5493e-04 [2.9139e-04 - 8.0125e-04] (2#)
    
```

### Performance
Code from [LayerTestBase.java:144](../../../../../../../../../MindsEye/src/test/java/com/simiacryptus/mindseye/layers/LayerTestBase.java#L144) executed in 0.12 seconds: 
```java
    getPerformanceTester().test(layer, inputPrototype);
```
Logging: 
```
    Evaluation performance: 3.9465 +- 0.5247 [3.4711 - 6.7739]
    Learning performance: 3.2959 +- 0.7113 [2.8583 - 8.8486]
    
```

