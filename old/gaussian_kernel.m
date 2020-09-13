function gaussian_kernel = gaussian_kernel(x1,x2,sigma)

gaussian_kernel = exp(-norm(x1-x2,2)/(2*sigma^2));
end
