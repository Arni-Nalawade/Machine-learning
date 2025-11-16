# main_test.py

from LearnedRec import LearnedRectangle

if __name__ == '__main__':
    # 1. Instantiate the class
    lr = LearnedRectangle()
    print("LearnedRectangle object instantiated.")

    # 2. Learn from a set of examples
    num_learning_examples = 50
    print(f"\nLearning from {num_learning_examples} examples...")
    lr.learn(num_learning_examples)
    
    # The boundaries should be learned from the positive examples in generate_example.py
    # From the sample `generate_example.py`, the first positive example is [15, 30]
    # Then it finds [10, 20] and updates the boundaries.
    # The next is [30, 50], and so on.
    # After 5 examples, the boundaries should be [10, 20] and [30, 50]
    print(f"Learned boundaries: {lr.learn}")
    
    # 3. Check the goodness of the learned rectangle
    num_tests = 20
    examples_per_test = 20
    error_threshold = 0.01
    
    print(f"\nChecking goodness with {num_tests} tests of {examples_per_test} examples each.")
    print(f"Error threshold (epsilon) is {error_threshold}.")
    
    # The `checkgoodness` method will use new examples from the generator
    # and count how many times the error rate exceeds the threshold.
    goodness_counter = lr.checkgoodness(num_tests, examples_per_test, error_threshold)
    
    print(f"The checkgoodness method returned: {goodness_counter}")
    
    # Let's manually trace the expected output with our sample `generate_example.py`:
    # The sequence of examples after learning will be (due to shuffling, it's not exact,
    # but the content will be from our `examples` list).
    #
    # Assume the generator gives:
    # Set 1: ([5, 25], False), ([35, 45], False), ([15, 30], True), ([25, 60], False)
    # Our rectangle is [10,30] x [20,50].
    # - Point [5, 25] is classified as False by our rectangle. Correct. Misclassifications: 0.
    # - Point [35, 45] is classified as False. Correct. Misclassifications: 0.
    # - Point [15, 30] is classified as True. Correct. Misclassifications: 0.
    # - Point [25, 60] is classified as False. Correct. Misclassifications: 0.
    # Misclassified count = 0. 0 > 4 * 0.2 is False. Counter does not increase.
    #
    # The actual output will depend on the shuffling, so the number of misclassifications
    # will vary. Run the code to see a specific result.
