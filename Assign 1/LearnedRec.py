from generate_example import generate_example

class LearnedRectangle:
    def __init__(self):
        # Your code here to initialize the object.
        self.boundaries = [] #Initialize tuple

    def learn(self, m):
        # Your code here to learn the target rectangle. You should
        # get the m examples by invoking next(generate_example()).
        # You can assume that m is an integer >= 1.
        next_point = generate_example()
        
        for i in range(m):
            point, b = next(next_point)
            num_dimensions = len(point)

            if i == 0:
                #Find boundaries
                self.boundaries = [(float('inf'), float('-inf'))] * num_dimensions

            if b: #If true (b is posisitve) then include the point
                for j in range(num_dimensions):
                    current_min, current_max = self.boundaries[j]
                    new_min = min(current_min, point[j])
                    new_max = max(current_max, point[j])
                    self.boundaries[j] = (new_min, new_max)

    def _classify(self, point):
        # Helper method to check if a point is inside the learned rectangle
        if not self.boundaries:
            return False # Error handling in case there are no positive points
            
        for j in range(len(point)):
            if not (self.boundaries[j][0] <= point[j] <= self.boundaries[j][1]): # checking if a positive point is within the bounds of the rectangle
                return False
        return True

    def checkgoodness(self, n, k, epsilon):
        # Your code here for the following, whose intent is to check
        # the goodness of your learned rectangle.
        # Initialize a counter, and perform the following n times.
        # For k examples, check whether the proportion of those k
        # that are misclassified by your learned rectangle is > epsilon.
        # If yes, increase your counter by 1.
        # Return the value of your counter.
        #
        # E.g., suppose n = 2, k = 5 and epsilon = 0.2. This means
        # you will consider 2 sets of 5 examples each. Suppose for the
        # first # set of 5 examples, your learned rectangle has
        # misclassified 1 of those 5 examples. As # 1 <= 5 x 0.2, you
        # do not increase your counter. Suppose for the second set of
        # 5 examples, your learned rectangle misclassied 3 out of 5.
        # As 3 > 5 x 0.2, you will increase your counter by 1. And you
        # will return 1, i.e., the value of the counter, as your result.
        next_point = generate_example()
        counter = 0

        for _ in range(n):
            misclassified_count = 0
            for _ in range(k):
                point, b = next(next_point)
                
                # Check if the learned rectangle's classification = example's
                is_positive = self._classify(point)
                
                if (is_positive and not b) or (not is_positive and b):
                    misclassified_count += 1
            
            # Check if num misclassified examples > epsilon
            if misclassified_count > k * epsilon:
                counter += 1

        return counter