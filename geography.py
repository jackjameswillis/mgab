import torch

class Ring:

    def __init__(self, population_size, device):

        self.population_size = population_size

        self.device = device

    def tournament(self, deme_size):

        deme_size -= 1

        start = torch.randint(0, self.population_size, (1,), device=self.device).item()

        selected = -torch.ones(self.population_size, device=self.device).to(torch.int)

        for i in range(self.population_size):

            shifted_indices = torch.arange(start + i, start + i + self.population_size, device=self.device) % self.population_size
                
            if selected[shifted_indices[0]] == -1:

                deme = shifted_indices[1:deme_size+1]

                deme = deme[selected[deme] == -1]

                select = deme[torch.randint(0, len(deme), (1,)).item()]

                selected[shifted_indices[0]] = select

                selected[select] = shifted_indices[0]
            
        return selected

class SmallWorld:
    """
    Small world topology for tournament selection.
    Each individual has a fixed local neighborhood defined by a Watts-Strogatz graph.
    The deme size is implicitly the number of neighbors (2*k).
    """

    def __init__(self, population_size, device, k=4, p=0.1):
        """
        Initialize small world topology.
        
        Args:
            population_size: Number of individuals in population
            device: PyTorch device (cpu or cuda)
            k: Number of neighbors on each side (total 2*k neighbors)
            p: Rewiring probability (0 = regular ring, 1 = random graph)
        """
        self.population_size = population_size
        self.device = device
        self.k = k
        self.p = p
        
        # Build adjacency matrix for small world topology
        self._build_topology()

    def _build_topology(self):
        """Build the small world adjacency structure."""
        # Start with regular ring (k neighbors on each side)
        neighbors = torch.zeros(self.population_size, 2 * self.k, dtype=torch.int, device=self.device)
        
        for i in range(self.population_size):
            for j in range(self.k):
                # Left neighbors
                neighbors[i, j] = (i - self.k + j) % self.population_size
                # Right neighbors
                neighbors[i, self.k + j] = (i + self.k - j) % self.population_size
        
        # Rewire with probability p (Watts-Strogatz)
        rewired = neighbors.clone()
        for i in range(self.population_size):
            for j in range(2 * self.k):
                if torch.rand(1, device=self.device).item() < self.p:
                    # Rewire to random node (avoid self-loops and duplicates)
                    new_target = torch.randint(0, self.population_size, (1,), device=self.device).item()
                    while new_target == i or new_target in neighbors[i]:
                        new_target = torch.randint(0, self.population_size, (1,), device=self.device).item()
                    rewired[i, j] = new_target
        
        self.neighbors = rewired

    def tournament(self, deme_size):
        """
        Perform tournament selection using the fixed small world topology.
        The deme consists of all neighbors defined by the topology.
        
        Returns:
            selected: Array mapping each position to selected individual

        deme_size does nothing here
        """
        start = torch.randint(0, self.population_size, (1,), device=self.device).item()
        
        selected = -torch.ones(self.population_size, device=self.device).to(torch.int)
        
        for i in range(self.population_size):
            current = (start + i) % self.population_size
            
            if selected[current] == -1:
                # Get all neighbors from small world topology
                deme = self.neighbors[current]
                
                # Filter out already selected individuals
                deme = deme[selected[deme] == -1]
                
                if len(deme) > 0:
                    select = deme[torch.randint(0, len(deme), (1,), device=self.device).item()]
                    selected[current] = select
                    selected[select] = current
        
        return selected