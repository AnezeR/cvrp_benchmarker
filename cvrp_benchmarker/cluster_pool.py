from typing import Type, Optional, Sequence, Union
from dask.distributed import LocalCluster, Client
from dask_jobqueue import JobQueueCluster


class ClusterPool:
    """Use this to manage paralellization capabilities of the benchmarker."""
    def __init__(
        self,
        clusters: Optional[Sequence[Union[LocalCluster, JobQueueCluster]]] = None,
        cluster_type: Optional[Union[Type[LocalCluster], Type[JobQueueCluster]]] = LocalCluster,
        n_clusters: Optional[int] = 1,
        workers_per_cluster: Optional[int] = 1,
        **kwargs
    ) -> None:
        """Create a cluster pool by either specifying clusters yourself or creating them automatically.
        Pass additional cluster arguments as kwargs.

        Args:
            clusters (Optional[Sequence[Union[LocalCluster, JobQueueCluster]]], optional): 
                A list of clusters to create a cluster pool from.
                Defaults to None.
            cluster_type (Optional[Union[Type[LocalCluster], Type[JobQueueCluster]]], optional): 
                A base cluster type for automatic creation of identic clusters.
                Read dask documentation to learn more
                at https://docs.dask.org/en/latest/deploying.html or https://jobqueue.dask.org/en/latest/examples.html.
                Defaults to LocalCluster.
            n_clusters (Optional[int], optional): Number of clusters for automatic creation.
                For best performance should be a divisor of number of runners times number of trials.
                Defaults to 1.
            workers_per_cluster (Optional[int], optional): Number of workers for a single cluster.
                For best performance should be a divisor of number of problems times number of trials times number of runners.
                Defaults to 1.
            All additional arguments are passed as Cluster's __init__ arguments.
        """
        if clusters is None:
            self.__clusters = []
            self.__clients = []
            for _ in range(n_clusters):
                if cluster_type is LocalCluster:
                    self.__clusters.append(cluster_type(kwargs, n_workers=workers_per_cluster, threads_per_worker=1))
                elif issubclass(cluster_type, JobQueueCluster):
                    self.__clusters.append(cluster_type(kwargs, n_workers=workers_per_cluster, cores=1))
            self.__clients = [Client(cluster) for cluster in self.__clusters]
        else:
            self.__clusters = clusters
        
        self.__current_cluster = 0
    
    def __enter__(self):
        return self
    
    def close(self):
        for cluster, client in zip(self.__clusters, self.__clients):
            client.close()
            cluster.close() 
    
    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.close()

    def next(self):
        return_cluster = self.__clients[self.__current_cluster]
        self.__current_cluster += 1
        self.__current_cluster %= len(self.__clusters)
        return return_cluster

    def size(self):
        return len(self.__clusters)
    