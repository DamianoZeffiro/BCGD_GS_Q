import heapq

class Heap:
    def __init__(self, gradient, f_priorities):
        self.f_priorities = f_priorities
        self.heap = [(-f_priorities(val, idx), idx) for idx, val in enumerate(gradient)]
        heapq.heapify(self.heap)
        self.heap = [(-vali[0], vali[1]) for vali in self.heap]
        self.dict = {vali[1]: i for (i, vali) in enumerate(self.heap)} # dictionary cointaining the index of each element in the heap

    def get_max(self):
        return self.heap[0]

    def update_priority(self, idx, new_val):
        new_priority = self.f_priorities(new_val, idx)
        i = self.dict[idx]
        old_priority, _ = self.heap[i]
        self.heap[i] = (new_priority, idx)
        if new_priority > old_priority:
            # New priority is higher than the old one, sift up
            while i > 0:
                parent_i = (i - 1) // 2
                if self.heap[parent_i] < self.heap[i]:
                    heap_parent_i = self.heap[parent_i]
                    self.heap[parent_i] = self.heap[i]
                    self.heap[i] = heap_parent_i
                    self.dict[self.heap[parent_i][1]], self.dict[self.heap[i][1]] = parent_i, i
                    i = parent_i
                else:
                    break
        else:
            # New priority is lower or equal to the old one, sift down
            while True:
                left_child_i = 2 * i + 1
                right_child_i = 2 * i + 2

                #TODO: implement the rest of the sift down


