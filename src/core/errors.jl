abstract type TiliaError <: Exception end

for error_type in (
    :InvalidHyperparameterError, :SchemaMismatchError, :UnsupportedDataError,
    :UnsupportedBackendError, :ConvergenceError, :NumericalFailureError,
    :PersistenceVersionError, :GraphValidationError, :LeakageError,
)
    @eval struct $error_type <: TiliaError
        message::String
    end
    @eval Base.showerror(io::IO, error::$error_type) = print(io, error.message)
end
