# Setup cluster workers for installed package
setup_cluster_workers_installed <- function(cl, pkg_name = "DirichletRF") {
  
  # Load Rcpp on workers
  parallel::clusterEvalQ(cl, library(Rcpp))
  
  # Load the package's shared library on each worker
  parallel::clusterCall(cl, function(pkg_name) {
    # Find the installed package
    lib_path <- find.package(pkg_name)
    
    # Load the compiled DLL
    dll_file <- file.path(lib_path, "libs", 
                          paste0(pkg_name, .Platform$dynlib.ext))
    
    if (file.exists(dll_file)) {
      library.dynam(pkg_name, pkg_name, lib_path)
      return("DLL loaded successfully")
    } else {
      return("DLL not found")
    }
  }, pkg_name)
  
  # Verify functions are available
  parallel::clusterCall(cl, function() {
    if (exists("DirichletForest") && exists("PredictDirichletForest")) {
      return("Functions available")
    } else {
      return("Functions not found")
    }
  })
}

# Setup cluster workers for development (using sourceCpp)
setup_cluster_workers <- function(cl, cpp_file = NULL) {
  
  # Load Rcpp on workers
  parallel::clusterEvalQ(cl, library(Rcpp))
  
  # Auto-detect cpp file if not provided
  if (is.null(cpp_file)) {
    # Try to find it relative to the package source
    possible_paths <- c(
      "src/dirichlet_forest.cpp",  # If running from package root
      #"../src/dirichlet_forest.cpp",  # If running from R/
      file.path(getwd(), "src/dirichlet_forest.cpp")  # Absolute from working dir
    )
    
    cpp_file <- NULL
    for (path in possible_paths) {
      if (file.exists(path)) {
        cpp_file <- normalizePath(path)
        break
      }
    }
    
    if (is.null(cpp_file)) {
      stop("Could not find dirichlet_forest.cpp. Please provide cpp_file parameter.")
    }
  }
  
  if (!file.exists(cpp_file)) {
    stop("C++ file not found: ", cpp_file)
  }
  
  message("Loading C++ functions from: ", cpp_file)
  
  # Source the C++ file on each worker
  parallel::clusterCall(cl, function(cpp_path) {
    Rcpp::sourceCpp(cpp_path)
    return("C++ functions loaded")
  }, cpp_path = cpp_file)
  
  # Verify functions are available
  result <- parallel::clusterCall(cl, function() {
    if (exists("DirichletForest") && exists("PredictDirichletForest")) {
      return("Functions available")
    } else {
      return("Functions not found")
    }
  })
  
  message("Worker setup results: ", paste(unlist(result), collapse = " "))
}