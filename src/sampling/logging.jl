"""
$(SIGNATURES)
Return `Progress` struct with arguments from `info` for sampling session.

# Examples
```julia
```

"""
function progress(report::ProgressReport, info::SampleInfo)
    return ProgressMeter.Progress(
        info.iterations * info.Nalgorithms * info.Nchains;
        enabled=report.bar,
        report.kwargs...,
    )
end

############################################################################################
function update!(
    progress::ProgressMeter.Progress, log::SilentLog, diagnostics::D
) where {D<:AbstractDiagnostics}
    return ProgressMeter.next!(progress)
end
function update!(
    progress::ProgressMeter.Progress, log::ConsoleLog, diagnostics::D
) where {D<:AbstractDiagnostics}
    return ProgressMeter.next!(
        progress; showvalues=BaytesCore.generate_showvalues(diagnostics)
    )
end

############################################################################################
#export
export progress, update!
