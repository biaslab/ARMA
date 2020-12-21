@naiveVariationalRule(:node_type     => AutoRegressiveMovingAverage,
                      :outbound_type => Message{GaussianMeanPrecision},
                      :inbound_types => (Nothing, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalARMAOutNPPPP)

@naiveVariationalRule(:node_type     => AutoRegressiveMovingAverage,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, Nothing, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalARMAIn1PNPPP)

@naiveVariationalRule(:node_type     => AutoRegressiveMovingAverage,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, Nothing, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalARMAIn2PPNPP)

@naiveVariationalRule(:node_type     => AutoRegressiveMovingAverage,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, Nothing, ProbabilityDistribution),
                      :name          => VariationalARMAIn3PPPNP)

@naiveVariationalRule(:node_type     => AutoRegressiveMovingAverage,
                      :outbound_type => Message{Gamma},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, Nothing),
                      :name          => VariationalARMAIn4PPPPN)