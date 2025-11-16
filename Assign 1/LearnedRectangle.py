from generate_example import generate_example

class LearnedRectangle:
    def __init__(self):
        # Initialize an empty list to store the boundaries of the rectangle
        # self.boundaries will be a list of tuples [(min, max), (min, max), ...],
        # one for each dimension.
        self.boundaries = []

    def learn(self, m):
        # The generator must be initialized once
        example_generator = generate_example()
        
        for i in range(m):
            point, b = next(example_generator)
            num_dimensions = len(point)

            if i == 0:
                # On the first example, discover the number of dimensions
                # and initialize the boundaries.
                self.boundaries = [(float('inf'), float('-inf'))] * num_dimensions

            if b: # If it's a positive example
                # Update the boundaries to include this point
                for j in range(num_dimensions):
                    current_min, current_max = self.boundaries[j]
                    new_min = min(current_min, point[j])
                    new_max = max(current_max, point[j])
                    self.boundaries[j] = (new_min, new_max)

    def _classify(self, point):
        # Helper method to check if a point is inside the learned rectangle
        if not self.boundaries:
            # If no positive examples were seen, the rectangle is undefined
            return False
            
        for j in range(len(point)):
            if not (self.boundaries[j][0] <= point[j] <= self.boundaries[j][1]):
                return False
        return True

    def checkgoodness(self, n, k, epsilon):
        example_generator = generate_example()
        counter = 0

        for _ in range(n):
            misclassified_count = 0
            for _ in range(k):
                point, b = next(example_generator)
                
                # Check if the learned rectangle's classification matches the example's
                is_positive = self._classify(point)
                
                # If there's a mismatch, increment the misclassified count
                if (is_positive and not b) or (not is_positive and b):
                    misclassified_count += 1
            
            # Check if the proportion of misclassified examples exceeds epsilon
            if misclassified_count > k * epsilon:
                counter += 1

        return counter